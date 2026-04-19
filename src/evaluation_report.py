"""
Evaluation report generation for arrhythmia detection models.

:class:`EvaluationReport` generates per-model reports (JSON metrics +
matplotlib figures) and a multi-model comparison report.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

_AAMI_CLASSES = ["N", "S", "V", "F", "Q"]


class EvaluationReport:
    """Generate and persist evaluation reports for baseline and DL models.

    Parameters
    ----------
    class_names:
        Class name strings used in all plots / tables.
    """

    def __init__(self, class_names: Optional[List[str]] = None) -> None:
        self.class_names = class_names or _AAMI_CLASSES
        self.evaluator = ModelEvaluator(class_names=self.class_names)

    # ------------------------------------------------------------------ #
    #  Baseline (sklearn) report                                           #
    # ------------------------------------------------------------------ #

    def generate_baseline_report(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_dir: str,
    ) -> Dict[str, Any]:
        """Generate a full evaluation report for a scikit-learn model.

        Parameters
        ----------
        model:
            A fitted :class:`~src.baseline_model.BaselineModel` instance.
        X_test:
            Test feature matrix.
        y_test:
            True integer labels.
        save_dir:
            Directory where figures and the JSON report are saved.

        Returns
        -------
        dict
            Report data (also written to ``save_dir/baseline_report.json``).
        """
        os.makedirs(save_dir, exist_ok=True)

        y_pred = model.predict(X_test)
        try:
            y_prob = model.predict_proba(X_test)
        except AttributeError:
            y_prob = None

        metrics = self.evaluator.compute_metrics(y_test, y_pred, y_prob)

        # Confusion matrix
        self.evaluator.plot_confusion_matrix(
            metrics["confusion_matrix"],
            save_path=os.path.join(save_dir, "baseline_confusion_matrix.png"),
        )

        # PR curves (if probabilities are available)
        if y_prob is not None:
            self.evaluator.plot_precision_recall_curves(
                y_test, y_prob,
                save_path=os.path.join(save_dir, "baseline_pr_curves.png"),
            )

        report_data = self._serialisable_metrics(metrics)
        self.save_report(report_data, os.path.join(save_dir, "baseline_report.json"))
        return report_data

    # ------------------------------------------------------------------ #
    #  Deep-learning report                                                #
    # ------------------------------------------------------------------ #

    def generate_dl_report(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        device: Any,
        save_dir: str,
        model_name: str = "dl_model",
    ) -> Dict[str, Any]:
        """Generate a full evaluation report for a PyTorch model.

        Parameters
        ----------
        model:
            Trained PyTorch model.
        test_loader:
            DataLoader for the test set.
        device:
            Torch device.
        save_dir:
            Output directory for figures and JSON.
        model_name:
            Prefix used for output file names.

        Returns
        -------
        dict
            Report data.
        """
        os.makedirs(save_dir, exist_ok=True)

        y_pred, y_true, y_prob = self.evaluator.evaluate(model, test_loader, device)
        metrics = self.evaluator.compute_metrics(y_true, y_pred, y_prob)

        # Confusion matrix
        self.evaluator.plot_confusion_matrix(
            metrics["confusion_matrix"],
            save_path=os.path.join(save_dir, f"{model_name}_confusion_matrix.png"),
        )

        # PR curves
        self.evaluator.plot_precision_recall_curves(
            y_true, y_prob,
            save_path=os.path.join(save_dir, f"{model_name}_pr_curves.png"),
        )

        report_data = self._serialisable_metrics(metrics)
        self.save_report(
            report_data,
            os.path.join(save_dir, f"{model_name}_report.json"),
        )
        return report_data

    # ------------------------------------------------------------------ #
    #  Comparison report                                                   #
    # ------------------------------------------------------------------ #

    def generate_comparison_report(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        save_dir: str,
    ) -> None:
        """Generate a bar-chart comparison across multiple model results.

        Parameters
        ----------
        results_dict:
            Mapping of ``{model_name: metrics_dict}`` where each
            ``metrics_dict`` has at least ``"macro_f1"`` and
            ``"weighted_f1"`` keys (as returned by the other report
            methods).
        save_dir:
            Output directory.
        """
        os.makedirs(save_dir, exist_ok=True)

        model_names = list(results_dict.keys())
        macro_f1s = [results_dict[m].get("macro_f1", 0.0) for m in model_names]
        weighted_f1s = [results_dict[m].get("weighted_f1", 0.0) for m in model_names]
        auc_rocs = [results_dict[m].get("auc_roc") or 0.0 for m in model_names]

        x = np.arange(len(model_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 2), 5))
        ax.bar(x - width, macro_f1s, width, label="Macro F1")
        ax.bar(x, weighted_f1s, width, label="Weighted F1")
        ax.bar(x + width, auc_rocs, width, label="AUC-ROC")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "model_comparison.png"), dpi=150)
        plt.close(fig)

        # Save combined JSON
        combined = {name: results_dict[name] for name in model_names}
        self.save_report(combined, os.path.join(save_dir, "comparison_report.json"))
        logger.info("Comparison report saved to %s.", save_dir)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save_report(self, report_data: Any, path: str) -> None:
        """Write *report_data* to a JSON file at *path*.

        Parameters
        ----------
        report_data:
            JSON-serialisable data (dict, list, etc.).
        path:
            Output file path (must end in ``.json``).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report_data, fh, indent=2, default=_json_default)
        logger.info("Report saved to %s", path)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _serialisable_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy arrays in *metrics* to plain Python types."""
        out: Dict[str, Any] = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, np.generic):
                out[k] = v.item()
            else:
                out[k] = v
        return out


def _json_default(obj: Any) -> Any:
    """JSON serialiser for numpy types and other non-standard objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable.")
