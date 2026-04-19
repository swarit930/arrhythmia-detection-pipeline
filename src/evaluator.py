"""
Patient-aware data splitting and model evaluation utilities.

Provides:
* :class:`PatientSplitter` – ensures **no patient overlap** between train /
  val / test splits.
* :class:`ModelEvaluator` – computes and visualises per-class metrics for
  both sklearn and PyTorch models.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.data_loader import ECGDataLoader

logger = logging.getLogger(__name__)


# ======================================================================= #
#  Patient-aware split                                                     #
# ======================================================================= #

class PatientSplitter:
    """Split MIT-BIH record lists without patient overlap across splits.

    Parameters
    ----------
    random_state:
        Seed for reproducible shuffling.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def split(
        self,
        record_list: List[str],
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Return (train_records, val_records, test_records).

        Algorithm
        ---------
        1. Extract patient IDs from record names (first 3 digits).
        2. Shuffle unique patients.
        3. Assign patients to test / val / train proportionally.
        4. Map patient IDs back to record lists.

        Parameters
        ----------
        record_list:
            Full list of record identifiers to split.
        test_size:
            Fraction of *patients* to assign to the test set.
        val_size:
            Fraction of *patients* to assign to the validation set.

        Returns
        -------
        train_records : List[str]
        val_records : List[str]
        test_records : List[str]
        """
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be in (0, 1), got {test_size}")
        if not 0 < val_size < 1:
            raise ValueError(f"val_size must be in (0, 1), got {val_size}")
        if test_size + val_size >= 1.0:
            raise ValueError("test_size + val_size must be < 1.0")

        patient_ids = ECGDataLoader.get_patient_ids(record_list)
        unique_patients = sorted(set(patient_ids))

        rng = np.random.RandomState(self.random_state)
        rng.shuffle(unique_patients)  # type: ignore[arg-type]

        n_patients = len(unique_patients)
        n_test = max(1, int(np.ceil(n_patients * test_size)))
        n_val = max(1, int(np.ceil(n_patients * val_size)))
        n_train = n_patients - n_test - n_val

        if n_train <= 0:
            raise ValueError(
                f"Not enough patients ({n_patients}) for the requested split sizes."
            )

        test_patients = set(unique_patients[:n_test])
        val_patients = set(unique_patients[n_test: n_test + n_val])
        train_patients = set(unique_patients[n_test + n_val:])

        logger.info(
            "Patient split: %d train / %d val / %d test patients.",
            len(train_patients), len(val_patients), len(test_patients),
        )

        train_records: List[str] = []
        val_records: List[str] = []
        test_records: List[str] = []

        for rec, pid in zip(record_list, patient_ids):
            if pid in test_patients:
                test_records.append(rec)
            elif pid in val_patients:
                val_records.append(rec)
            else:
                train_records.append(rec)

        logger.info(
            "Record split: %d train / %d val / %d test records.",
            len(train_records), len(val_records), len(test_records),
        )
        return train_records, val_records, test_records


# ======================================================================= #
#  Model evaluator                                                         #
# ======================================================================= #

class ModelEvaluator:
    """Compute and visualise performance metrics for a trained PyTorch model.

    Parameters
    ----------
    class_names:
        Human-readable names for each class index (e.g. ``["N","S","V","F","Q"]``).
    """

    def __init__(self, class_names: Optional[List[str]] = None) -> None:
        self.class_names = class_names or ["N", "S", "V", "F", "Q"]

    # ------------------------------------------------------------------ #
    #  Inference pass                                                      #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device | str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference on *dataloader* and collect predictions.

        Parameters
        ----------
        model:
            Trained PyTorch model.
        dataloader:
            DataLoader yielding ``(inputs, targets)`` batches.
        device:
            Computation device.

        Returns
        -------
        y_pred : np.ndarray, shape (n,)
        y_true : np.ndarray, shape (n,)
        y_prob : np.ndarray, shape (n, n_classes)
        """
        dev = torch.device(device) if isinstance(device, str) else device
        model.eval()
        model.to(dev)

        all_preds: List[np.ndarray] = []
        all_true: List[np.ndarray] = []
        all_prob: List[np.ndarray] = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(dev), batch[1]
                else:
                    inputs = batch["input"].to(dev)
                    targets = batch["label"]

                logits = model(inputs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_preds.append(preds)
                all_true.append(targets.cpu().numpy())
                all_prob.append(probs)

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_true)
        y_prob = np.concatenate(all_prob)
        return y_pred, y_true, y_prob

    # ------------------------------------------------------------------ #
    #  Metrics                                                             #
    # ------------------------------------------------------------------ #

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict:
        """Compute a rich set of classification metrics.

        Parameters
        ----------
        y_true:
            Ground-truth integer labels.
        y_pred:
            Predicted integer labels.
        y_prob:
            Predicted probabilities ``(n, n_classes)``; required for AUC.

        Returns
        -------
        dict with keys:
          ``confusion_matrix``, ``classification_report``,
          ``f1_per_class``, ``macro_f1``, ``weighted_f1``,
          ``sensitivity_per_class``, ``specificity_per_class``,
          ``ppv_per_class``, ``auc_roc`` (if *y_prob* provided).
        """
        cm = confusion_matrix(y_true, y_pred)
        report_str = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0,
        )
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Per-class sensitivity (recall), specificity, PPV
        n_classes = cm.shape[0]
        sensitivity = np.zeros(n_classes)
        specificity = np.zeros(n_classes)
        ppv = np.zeros(n_classes)

        for c in range(n_classes):
            tp = cm[c, c]
            fn = cm[c, :].sum() - tp
            fp = cm[:, c].sum() - tp
            tn = cm.sum() - tp - fn - fp

            sensitivity[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity[c] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ppv[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        result: Dict = {
            "confusion_matrix": cm,
            "classification_report": report_str,
            "f1_per_class": {self.class_names[i]: float(f1_per_class[i]) for i in range(len(f1_per_class))},
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "sensitivity_per_class": {self.class_names[i]: float(sensitivity[i]) for i in range(n_classes)},
            "specificity_per_class": {self.class_names[i]: float(specificity[i]) for i in range(n_classes)},
            "ppv_per_class": {self.class_names[i]: float(ppv[i]) for i in range(n_classes)},
        }

        # AUC-ROC (one-vs-rest, macro)
        if y_prob is not None:
            try:
                auc_roc = roc_auc_score(
                    y_true, y_prob,
                    multi_class="ovr",
                    average="macro",
                    labels=list(range(n_classes)),
                )
                result["auc_roc"] = float(auc_roc)
            except Exception as exc:
                logger.warning("Could not compute AUC-ROC: %s", exc)
                result["auc_roc"] = None

        logger.info(
            "Evaluation – macro_f1=%.4f, weighted_f1=%.4f",
            macro_f1, weighted_f1,
        )
        return result

    # ------------------------------------------------------------------ #
    #  Visualisation                                                       #
    # ------------------------------------------------------------------ #

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot and optionally save a normalised confusion matrix.

        Parameters
        ----------
        cm:
            Square integer confusion matrix from :func:`~sklearn.metrics.confusion_matrix`.
        class_names:
            Class label strings.  Defaults to ``self.class_names``.
        save_path:
            If provided, the figure is saved here (PNG/PDF).
        """
        names = class_names or self.class_names
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=names,
            yticklabels=names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Normalised Confusion Matrix")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150)
            logger.info("Confusion matrix saved to %s", save_path)
        plt.close(fig)

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot precision-recall curves for each class.

        Parameters
        ----------
        y_true:
            Ground-truth labels.
        y_prob:
            Predicted probabilities ``(n, n_classes)``.
        class_names:
            Class label strings.
        save_path:
            Optional path to save the figure.
        """
        names = class_names or self.class_names
        n_classes = y_prob.shape[1]

        fig, ax = plt.subplots(figsize=(8, 6))
        for c in range(n_classes):
            binary_true = (y_true == c).astype(int)
            prec, rec, _ = precision_recall_curve(binary_true, y_prob[:, c])
            area = auc(rec, prec)
            ax.plot(rec, prec, label=f"{names[c]} (AUC={area:.2f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves (one-vs-rest)")
        ax.legend(loc="best")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150)
            logger.info("PR curve saved to %s", save_path)
        plt.close(fig)
