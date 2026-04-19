"""
Real-time inference engine for arrhythmia detection.

:class:`InferenceEngine` wraps a trained CNN or LSTM model and exposes a
simple API for single-beat and full-record prediction, with optional
gradient-based explanation.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.preprocessor import ECGPreprocessor
from src.explainability import GradientExplainer

logger = logging.getLogger(__name__)

_AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
_DEFAULT_FS = 360


class InferenceEngine:
    """Load a trained model and perform beat-level arrhythmia classification.

    Parameters
    ----------
    model_path:
        Path to a ``.pth`` checkpoint file produced by
        :class:`~src.trainer.Trainer`.
    model_type:
        Architecture to instantiate: ``"cnn"`` or ``"lstm"``.
    device:
        Torch device string (``"cpu"`` or ``"cuda"``).  Auto-detected if
        ``None``.
    num_classes:
        Number of output classes (default 5).
    beat_len:
        Samples per beat (default 129).
    """

    def __init__(
        self,
        model_path: str,
        model_type: Literal["cnn", "lstm"] = "cnn",
        device: Optional[str] = None,
        num_classes: int = 5,
        beat_len: int = 129,
    ) -> None:
        self.model_path = model_path
        self.model_type = model_type
        self.num_classes = num_classes
        self.beat_len = beat_len
        self.class_names = _AAMI_CLASSES[:num_classes]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model: Optional[nn.Module] = None
        self._explainer: Optional[GradientExplainer] = None

    # ------------------------------------------------------------------ #
    #  Model loading                                                       #
    # ------------------------------------------------------------------ #

    def load_model(self) -> None:
        """Instantiate and load weights into the model.

        Raises
        ------
        FileNotFoundError
            If *model_path* does not exist.
        ValueError
            If *model_type* is not ``"cnn"`` or ``"lstm"``.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        if self.model_type == "cnn":
            from src.models_pytorch import ECG_CNN
            self.model = ECG_CNN(num_classes=self.num_classes, input_length=self.beat_len)
        elif self.model_type == "lstm":
            from src.models_pytorch import ECG_LSTM
            self.model = ECG_LSTM(num_classes=self.num_classes, beat_len=self.beat_len)
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'. Use 'cnn' or 'lstm'.")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model (%s) loaded from %s.", self.model_type.upper(), self.model_path)

    # ------------------------------------------------------------------ #
    #  Signal pre-processing                                               #
    # ------------------------------------------------------------------ #

    def preprocess_signal(
        self,
        signal: np.ndarray,
        fs: int = _DEFAULT_FS,
    ) -> np.ndarray:
        """Run the full ECG pre-processing chain on a raw signal.

        Parameters
        ----------
        signal:
            Raw 1-D (or 2-D multi-lead) ECG signal.
        fs:
            Sampling frequency in Hz.

        Returns
        -------
        np.ndarray
            Pre-processed 1-D signal (float32).
        """
        return ECGPreprocessor.preprocess_record(signal, fs=fs)

    # ------------------------------------------------------------------ #
    #  Single-beat prediction                                              #
    # ------------------------------------------------------------------ #

    def predict_beat(
        self,
        beat: np.ndarray,
    ) -> Tuple[int, float, np.ndarray]:
        """Classify a single pre-processed beat waveform.

        Parameters
        ----------
        beat:
            1-D array of *beat_len* samples.

        Returns
        -------
        predicted_class : int
        confidence : float  (max probability)
        probabilities : np.ndarray, shape ``(num_classes,)``
        """
        self._check_loaded()
        tensor = self._beat_to_tensor(beat)
        with torch.no_grad():
            logits = self.model(tensor)  # type: ignore[misc]
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        return predicted_class, confidence, probs

    # ------------------------------------------------------------------ #
    #  Full-record prediction                                              #
    # ------------------------------------------------------------------ #

    def predict_record(
        self,
        record_path: str,
        fs: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load, pre-process, segment, and classify every beat in a record.

        Parameters
        ----------
        record_path:
            Path (without extension) to a wfdb record (``.dat`` + ``.hea``).
        fs:
            Sampling frequency.  If ``None``, read from the record header.

        Returns
        -------
        pd.DataFrame
            Columns: ``beat_idx``, ``r_peak_sample``, ``predicted_class``,
            ``class_name``, ``confidence``, plus one column per AAMI class
            containing probabilities.
        """
        import wfdb

        self._check_loaded()

        record = wfdb.rdrecord(record_path)
        signal = record.p_signal
        actual_fs: int = fs or record.fs

        # Pre-process (use first lead)
        clean_signal = self.preprocess_signal(signal, actual_fs)
        r_peaks = ECGPreprocessor.detect_r_peaks(clean_signal, actual_fs)
        beats, valid_idx = ECGPreprocessor.segment_beats(clean_signal, r_peaks, actual_fs)

        rows: List[Dict] = []
        for i, (beat, vi) in enumerate(zip(beats, valid_idx)):
            pred_class, conf, probs = self.predict_beat(beat)
            row: Dict = {
                "beat_idx": i,
                "r_peak_sample": int(r_peaks[vi]),
                "predicted_class": pred_class,
                "class_name": self.class_names[pred_class],
                "confidence": conf,
            }
            for c, name in enumerate(self.class_names):
                row[f"prob_{name}"] = float(probs[c])
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Prediction + explanation                                            #
    # ------------------------------------------------------------------ #

    def predict_with_explanation(
        self,
        beat: np.ndarray,
        target_class: Optional[int] = None,
        n_steps: int = 50,
    ) -> Tuple[int, float, np.ndarray, np.ndarray]:
        """Classify a beat and compute Integrated Gradients attributions.

        Parameters
        ----------
        beat:
            1-D beat waveform array.
        target_class:
            Class to explain; defaults to the predicted class.
        n_steps:
            Riemann steps for Integrated Gradients.

        Returns
        -------
        predicted_class : int
        confidence : float
        probabilities : np.ndarray
        attributions : np.ndarray  (same length as *beat*)
        """
        self._check_loaded()

        pred_class, confidence, probs = self.predict_beat(beat)
        explain_class = target_class if target_class is not None else pred_class

        if self._explainer is None:
            self._explainer = GradientExplainer(self.model, self.device)  # type: ignore[arg-type]

        tensor = self._beat_to_tensor(beat)
        attributions = self._explainer.compute_integrated_gradients(
            tensor, explain_class, n_steps=n_steps
        )
        return pred_class, confidence, probs, attributions

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _check_loaded(self) -> None:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

    def _beat_to_tensor(self, beat: np.ndarray) -> torch.Tensor:
        """Convert a 1-D beat array to a ``(1, 1, L)`` tensor for CNN input."""
        arr = beat.astype(np.float32)
        # Pad or truncate to expected length
        if len(arr) < self.beat_len:
            arr = np.pad(arr, (0, self.beat_len - len(arr)))
        elif len(arr) > self.beat_len:
            arr = arr[: self.beat_len]
        tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        return tensor.to(self.device)
