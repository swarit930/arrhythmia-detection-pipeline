"""
Custom loss functions for imbalanced ECG beat classification.

Provides:
* :class:`FocalLoss` – down-weights easy examples so training focuses on
  hard / minority-class samples (Lin et al., 2017).
* :class:`WeightedCrossEntropyLoss` – standard cross-entropy with per-class
  inverse-frequency weights.
* :func:`compute_class_weights` – computes inverse-frequency weights from a
  label array.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ======================================================================= #
#  Focal Loss                                                              #
# ======================================================================= #

class FocalLoss(nn.Module):
    """Focal loss for multi-class classification (Lin et al., 2017).

    ``FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)``

    Parameters
    ----------
    alpha:
        Weighting scalar (float) applied to all classes equally, or a
        1-D :class:`torch.Tensor` of per-class weights.
    gamma:
        Focusing parameter (default 2.0).  Higher values reduce the
        relative loss for well-classified examples.
    reduction:
        ``"mean"`` (default) | ``"sum"`` | ``"none"``.
    """

    def __init__(
        self,
        alpha: float | torch.Tensor = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if isinstance(alpha, (int, float)):
            self.alpha: Optional[torch.Tensor] = None
            self._alpha_scalar: float = float(alpha)
        else:
            self.register_buffer("alpha", alpha.float())
            self._alpha_scalar = 1.0
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        inputs : torch.Tensor, shape ``(N, C)``
            Raw class logits.
        targets : torch.Tensor, shape ``(N,)``
            Ground-truth integer class indices.

        Returns
        -------
        torch.Tensor
            Scalar loss (or per-sample tensor if ``reduction="none"``).
        """
        log_probs = F.log_softmax(inputs, dim=1)               # (N, C)
        probs = torch.exp(log_probs)                            # (N, C)

        # Gather log-prob and prob of the true class
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)   # (N,)
        pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)           # (N,)

        # Focusing factor
        focal_weight = (1.0 - pt) ** self.gamma

        # Alpha weighting
        if hasattr(self, "alpha") and self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
        else:
            alpha_t = self._alpha_scalar

        loss = -alpha_t * focal_weight * log_pt                 # (N,)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ======================================================================= #
#  Weighted Cross-Entropy Loss                                             #
# ======================================================================= #

class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with fixed per-class weights.

    Parameters
    ----------
    class_weights:
        1-D tensor of per-class weights, e.g. from
        :func:`compute_class_weights`.
    reduction:
        ``"mean"`` (default) | ``"sum"`` | ``"none"``.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights.float())
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy loss.

        Parameters
        ----------
        inputs : torch.Tensor, shape ``(N, C)``
            Raw logits.
        targets : torch.Tensor, shape ``(N,)``
            Integer ground-truth labels.

        Returns
        -------
        torch.Tensor – scalar loss.
        """
        return F.cross_entropy(
            inputs,
            targets,
            weight=self.class_weights.to(inputs.device),
            reduction=self.reduction,
        )


# ======================================================================= #
#  Class weight computation                                                #
# ======================================================================= #

def compute_class_weights(
    labels: np.ndarray | torch.Tensor,
    num_classes: Optional[int] = None,
    smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute inverse-frequency class weights.

    ``weight_c = total_samples / (num_classes * count_c)``

    Weights are normalised so that their mean equals 1.

    Parameters
    ----------
    labels:
        1-D array / tensor of integer class labels.
    num_classes:
        Number of classes.  If ``None``, inferred from *labels*.
    smoothing:
        Add *smoothing* to all counts before computing weights (Laplace
        smoothing to handle absent classes gracefully).

    Returns
    -------
    torch.Tensor, shape ``(num_classes,)``
    """
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy().astype(int)
    else:
        labels_np = np.asarray(labels, dtype=int)

    if num_classes is None:
        num_classes = int(labels_np.max()) + 1

    counts = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        counts[c] = float(np.sum(labels_np == c)) + smoothing

    total = float(len(labels_np)) + smoothing * num_classes
    weights = total / (num_classes * counts)

    # Normalise so that mean weight == 1
    weights = weights / weights.mean()

    logger.debug("Class weights: %s", weights.round(3).tolist())
    return torch.tensor(weights, dtype=torch.float32)
