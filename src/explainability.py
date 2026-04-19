"""
Explainability utilities for ECG classification models.

Provides:
* :class:`GradientExplainer` – Integrated Gradients and saliency maps
  for CNN models using *captum*.
* :class:`LSTMAttentionVisualizer` – visualises the attention weights
  produced by :class:`~src.models_pytorch.ECG_LSTM`.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ======================================================================= #
#  GradientExplainer                                                       #
# ======================================================================= #

class GradientExplainer:
    """Compute gradient-based attributions for a CNN ECG classifier.

    Uses *captum*'s :class:`~captum.attr.IntegratedGradients` and
    :class:`~captum.attr.Saliency` under the hood.

    Parameters
    ----------
    model:
        Trained PyTorch model (e.g. :class:`~src.models_pytorch.ECG_CNN`).
    device:
        Torch device on which attribution is computed.
    """

    def __init__(self, model: nn.Module, device: torch.device | str = "cpu") -> None:
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------ #
    #  Integrated Gradients                                                #
    # ------------------------------------------------------------------ #

    def compute_integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        n_steps: int = 50,
    ) -> np.ndarray:
        """Compute Integrated Gradients attributions.

        Parameters
        ----------
        input_tensor:
            Input of shape ``(1, 1, L)`` (single beat, channel-first).
        target_class:
            Class index to explain.
        n_steps:
            Number of Riemann-sum approximation steps.

        Returns
        -------
        np.ndarray, shape ``(L,)``
            Attribution values per sample point.
        """
        try:
            from captum.attr import IntegratedGradients  # type: ignore

            ig = IntegratedGradients(self.model)
            baseline = torch.zeros_like(input_tensor).to(self.device)
            inp = input_tensor.to(self.device).requires_grad_(True)
            attrs = ig.attribute(
                inp,
                baselines=baseline,
                target=target_class,
                n_steps=n_steps,
            )
            return attrs.squeeze().detach().cpu().numpy()
        except ImportError:
            logger.warning("captum not installed; falling back to gradient saliency.")
            return self.compute_saliency_map(input_tensor, target_class)
        except Exception as exc:
            logger.error("Integrated gradients failed: %s", exc)
            return np.zeros(input_tensor.shape[-1])

    # ------------------------------------------------------------------ #
    #  Saliency map                                                        #
    # ------------------------------------------------------------------ #

    def compute_saliency_map(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """Compute a vanilla gradient saliency map.

        Parameters
        ----------
        input_tensor:
            Input of shape ``(1, 1, L)``.
        target_class:
            Class index to explain.

        Returns
        -------
        np.ndarray, shape ``(L,)``
            Absolute gradient values as the saliency map.
        """
        inp = input_tensor.to(self.device).requires_grad_(True)
        self.model.zero_grad()
        logits = self.model(inp)
        score = logits[0, target_class]
        score.backward()
        saliency = inp.grad.abs().squeeze().detach().cpu().numpy()
        return saliency

    # ------------------------------------------------------------------ #
    #  Visualisation                                                       #
    # ------------------------------------------------------------------ #

    def visualize_explanation(
        self,
        signal: np.ndarray,
        attributions: np.ndarray,
        predicted_class: int,
        confidence: float,
        class_names: Optional[list] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot ECG signal overlaid with attribution heatmap.

        Parameters
        ----------
        signal:
            1-D ECG beat waveform.
        attributions:
            Attribution values of the same length as *signal*.
        predicted_class:
            Integer index of the predicted class.
        confidence:
            Model confidence (probability) for *predicted_class*.
        class_names:
            Optional list of class name strings.
        save_path:
            If provided, save the figure here.
        """
        names = class_names or ["N", "S", "V", "F", "Q"]
        class_label = names[predicted_class] if predicted_class < len(names) else str(predicted_class)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        t = np.arange(len(signal))
        ax1.plot(t, signal, color="steelblue", linewidth=1.5)
        ax1.set_ylabel("Amplitude (a.u.)")
        ax1.set_title(f"ECG Beat – Predicted: {class_label} (conf={confidence:.2f})")

        # Attribution as bar chart coloured by sign
        colours = np.where(attributions >= 0, "crimson", "royalblue")
        ax2.bar(t, attributions, color=colours, width=1.0)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_ylabel("Attribution")
        ax2.set_xlabel("Sample index")

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150)
            logger.info("Explanation saved to %s", save_path)
        plt.close(fig)


# ======================================================================= #
#  LSTMAttentionVisualizer                                                 #
# ======================================================================= #

class LSTMAttentionVisualizer:
    """Visualise attention weights from :class:`~src.models_pytorch.ECG_LSTM`.

    Parameters
    ----------
    model:
        Trained :class:`~src.models_pytorch.ECG_LSTM` instance.
    device:
        Torch device.
    """

    def __init__(self, model: nn.Module, device: torch.device | str = "cpu") -> None:
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------ #
    #  Extract weights                                                     #
    # ------------------------------------------------------------------ #

    def get_attention_weights(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Return attention weights for a batch of beat sequences.

        Parameters
        ----------
        input_tensor:
            Shape ``(batch, seq_len, beat_len)``.

        Returns
        -------
        np.ndarray, shape ``(batch, seq_len)``
        """
        with torch.no_grad():
            inp = input_tensor.to(self.device)
            weights = self.model.get_attention_weights(inp)
        return weights.cpu().numpy()

    # ------------------------------------------------------------------ #
    #  Visualisation                                                       #
    # ------------------------------------------------------------------ #

    def visualize_attention(
        self,
        beats: np.ndarray,
        attention_weights: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot beat waveforms annotated with attention intensity.

        Parameters
        ----------
        beats:
            Array of shape ``(seq_len, beat_len)`` – one row per beat.
        attention_weights:
            1-D array of length ``seq_len`` with normalised attention values.
        save_path:
            Optional figure save path.
        """
        seq_len = len(beats)
        fig, axes = plt.subplots(seq_len, 1, figsize=(10, seq_len * 1.5), sharex=False)
        if seq_len == 1:
            axes = [axes]

        cmap = plt.get_cmap("Reds")
        w_norm = attention_weights / (attention_weights.max() + 1e-9)

        for i, (beat, w, ax) in enumerate(zip(beats, w_norm, axes)):
            color = cmap(0.3 + 0.7 * float(w))
            ax.plot(beat, color=color, linewidth=1.5)
            ax.set_ylabel(f"Beat {i}\nattn={float(attention_weights[i]):.3f}", fontsize=7)
            ax.tick_params(labelbottom=False, bottom=False)

        axes[-1].tick_params(labelbottom=True, bottom=True)
        axes[-1].set_xlabel("Sample index")
        fig.suptitle("LSTM Attention over Beat Sequence", fontsize=12)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150)
            logger.info("Attention plot saved to %s", save_path)
        plt.close(fig)
