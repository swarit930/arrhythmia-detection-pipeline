"""
ECG signal pre-processing utilities.

:class:`ECGPreprocessor` implements the full pre-processing chain required
before feature extraction or deep-learning inference:

1. Bandpass filter (0.5–45 Hz)
2. Baseline-wander removal via dual median filter
3. Z-score normalisation
4. R-peak detection (biosppy with Pan-Tompkins fallback)
5. Beat segmentation centred on each R-peak
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

logger = logging.getLogger(__name__)

# Default parameters (kept in sync with config.py constants)
_DEFAULT_FS: int = 360
_DEFAULT_LOW: float = 0.5
_DEFAULT_HIGH: float = 45.0
_DEFAULT_WINDOW_MS: int = 360  # → 129 samples at 360 Hz


class ECGPreprocessor:
    """Complete ECG pre-processing pipeline for single-lead signals.

    All methods are static so the class can be used without instantiation,
    but an instance-based workflow is also supported.
    """

    # ------------------------------------------------------------------ #
    #  Step 1 – Bandpass filter                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def bandpass_filter(
        signal: np.ndarray,
        fs: int = _DEFAULT_FS,
        low: float = _DEFAULT_LOW,
        high: float = _DEFAULT_HIGH,
        order: int = 4,
    ) -> np.ndarray:
        """Apply a zero-phase Butterworth bandpass filter.

        Parameters
        ----------
        signal:
            1-D array of ECG samples.
        fs:
            Sampling frequency in Hz.
        low:
            High-pass corner frequency (Hz).
        high:
            Low-pass corner frequency (Hz).
        order:
            Filter order (applied twice due to *filtfilt*).

        Returns
        -------
        np.ndarray
            Filtered signal with the same length as *signal*.
        """
        nyq = 0.5 * fs
        low_n = low / nyq
        high_n = high / nyq
        b, a = sp_signal.butter(order, [low_n, high_n], btype="band")
        return sp_signal.filtfilt(b, a, signal).astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Step 2 – Baseline wander removal                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def remove_baseline_wander(
        signal: np.ndarray,
        fs: int = _DEFAULT_FS,
    ) -> np.ndarray:
        """Remove baseline wander using dual median filtering.

        Two sequential median filters (200 ms and 600 ms windows) estimate
        the baseline trend; subtracting it removes low-frequency wander.

        Parameters
        ----------
        signal:
            1-D ECG array.
        fs:
            Sampling frequency in Hz.

        Returns
        -------
        np.ndarray
            Signal with baseline wander removed.
        """
        # Window lengths must be odd
        win1 = _odd(int(0.200 * fs))  # 200 ms
        win2 = _odd(int(0.600 * fs))  # 600 ms
        baseline = sp_signal.medfilt(sp_signal.medfilt(signal.astype(np.float64), win1), win2)
        return (signal.astype(np.float64) - baseline).astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Step 3 – Normalisation                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def normalize_signal(signal: np.ndarray) -> np.ndarray:
        """Z-score normalise a signal to zero mean and unit variance.

        Parameters
        ----------
        signal:
            1-D or N-D array of signal samples.

        Returns
        -------
        np.ndarray
            Normalised array of the same shape.
        """
        std = np.std(signal)
        if std < 1e-8:
            logger.warning("Signal has near-zero standard deviation; returning zeros.")
            return np.zeros_like(signal, dtype=np.float32)
        return ((signal - np.mean(signal)) / std).astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Step 4 – R-peak detection                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_r_peaks(
        signal: np.ndarray,
        fs: int = _DEFAULT_FS,
    ) -> np.ndarray:
        """Detect R-peak positions in a pre-processed ECG signal.

        Tries *biosppy* first; falls back to a simple Pan-Tompkins-style
        derivative-energy detector if biosppy is unavailable or fails.

        Parameters
        ----------
        signal:
            Pre-processed 1-D ECG signal.
        fs:
            Sampling frequency in Hz.

        Returns
        -------
        np.ndarray
            Array of R-peak sample indices (int64).
        """
        # --- primary: biosppy ---
        try:
            from biosppy.signals import ecg as biosppy_ecg  # type: ignore

            out = biosppy_ecg.ecg(signal=signal.astype(np.float64), sampling_rate=fs, show=False)
            r_peaks: np.ndarray = out["rpeaks"]
            if len(r_peaks) > 0:
                logger.debug("biosppy detected %d R-peaks.", len(r_peaks))
                return r_peaks.astype(np.int64)
        except Exception as exc:
            logger.warning("biosppy R-peak detection failed (%s); using fallback.", exc)

        # --- fallback: Pan-Tompkins-style detector ---
        return ECGPreprocessor._pan_tompkins_detector(signal, fs)

    @staticmethod
    def _pan_tompkins_detector(signal: np.ndarray, fs: int) -> np.ndarray:
        """Simplified Pan-Tompkins R-peak detector used as a fallback.

        Steps: differentiate → square → integrate → threshold → find peaks.
        """
        # 1. Differentiate
        diff = np.diff(signal.astype(np.float64), prepend=signal[0])
        # 2. Square
        squared = diff ** 2
        # 3. Moving window integration (~150 ms window)
        win = max(1, int(0.150 * fs))
        kernel = np.ones(win) / win
        integrated = np.convolve(squared, kernel, mode="same")
        # 4. Adaptive threshold (mean of the integrated signal)
        threshold = np.mean(integrated) * 0.5
        above = (integrated > threshold).astype(int)
        # 5. Find rising edges and pick the max within each pulse
        edges = np.diff(above, prepend=0)
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        if len(ends) < len(starts):
            ends = np.append(ends, len(signal) - 1)

        r_peaks: List[int] = []
        min_rr = int(0.2 * fs)  # minimum 200 ms between beats
        last_peak = -min_rr
        for s, e in zip(starts, ends):
            idx = s + int(np.argmax(signal[s:e + 1]))
            if idx - last_peak >= min_rr:
                r_peaks.append(idx)
                last_peak = idx

        return np.array(r_peaks, dtype=np.int64)

    # ------------------------------------------------------------------ #
    #  Step 5 – Beat segmentation                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def segment_beats(
        signal: np.ndarray,
        r_peaks: np.ndarray,
        fs: int = _DEFAULT_FS,
        window_ms: int = _DEFAULT_WINDOW_MS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract fixed-length windows centred on R-peaks.

        Parameters
        ----------
        signal:
            Pre-processed 1-D ECG signal.
        r_peaks:
            Array of R-peak sample indices.
        fs:
            Sampling frequency in Hz.
        window_ms:
            Total window length in milliseconds (default 360 ms → 129 samples).

        Returns
        -------
        beats : np.ndarray, shape (n_valid_beats, window_samples)
            Each row is one beat waveform.
        valid_indices : np.ndarray, shape (n_valid_beats,)
            Indices into *r_peaks* for beats that fit within the signal.
        """
        half = int(window_ms / 2 * fs / 1000)  # 64 samples
        total = 2 * half + 1                    # 129 samples
        n = len(signal)

        beats: List[np.ndarray] = []
        valid_indices: List[int] = []

        for i, rp in enumerate(r_peaks):
            start = int(rp) - half
            end = start + total
            if start < 0 or end > n:
                continue  # skip beats too close to signal boundaries
            beats.append(signal[start:end].astype(np.float32))
            valid_indices.append(i)

        if len(beats) == 0:
            return np.empty((0, total), dtype=np.float32), np.array([], dtype=np.int64)

        return np.stack(beats), np.array(valid_indices, dtype=np.int64)

    # ------------------------------------------------------------------ #
    #  High-level helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def preprocess_record(
        signal: np.ndarray,
        fs: int = _DEFAULT_FS,
        lead: int = 0,
    ) -> np.ndarray:
        """Apply the full pre-processing chain to one ECG record.

        Steps: select lead → bandpass filter → baseline removal → normalise.

        Parameters
        ----------
        signal:
            2-D array shaped ``(n_samples, n_leads)`` **or** 1-D array.
        fs:
            Sampling frequency in Hz.
        lead:
            Lead index to use if *signal* is multi-lead.

        Returns
        -------
        np.ndarray
            Pre-processed 1-D signal.
        """
        if signal.ndim == 2:
            sig_1d = signal[:, lead].astype(np.float32)
        else:
            sig_1d = signal.astype(np.float32)

        sig_1d = ECGPreprocessor.bandpass_filter(sig_1d, fs)
        sig_1d = ECGPreprocessor.remove_baseline_wander(sig_1d, fs)
        sig_1d = ECGPreprocessor.normalize_signal(sig_1d)
        return sig_1d

    @staticmethod
    def extract_beats(
        signal: np.ndarray,
        r_peaks: np.ndarray,
        labels: np.ndarray,
        fs: int = _DEFAULT_FS,
        window_ms: int = _DEFAULT_WINDOW_MS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment beats and return aligned (beats, labels) arrays.

        Parameters
        ----------
        signal:
            Pre-processed 1-D ECG signal.
        r_peaks:
            Array of R-peak indices (must align with *labels*).
        labels:
            Integer label for each R-peak.
        fs:
            Sampling frequency in Hz.
        window_ms:
            Beat window in milliseconds.

        Returns
        -------
        beats : np.ndarray, shape (n_valid, window_samples)
        labels_out : np.ndarray, shape (n_valid,)
        """
        beats, valid_idx = ECGPreprocessor.segment_beats(signal, r_peaks, fs, window_ms)
        labels_out = labels[valid_idx]
        return beats, labels_out


# ------------------------------------------------------------------ #
#  Module-level helpers                                               #
# ------------------------------------------------------------------ #

def _odd(n: int) -> int:
    """Return *n* if odd, else *n* + 1 (scipy medfilt requires odd kernels)."""
    return n if n % 2 == 1 else n + 1
