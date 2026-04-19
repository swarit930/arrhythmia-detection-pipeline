"""
Scikit-learn baseline model for arrhythmia classification.

:class:`BaselineModel` wraps a scikit-learn ``Pipeline`` (StandardScaler +
classifier) and exposes train / predict / evaluate / save / load helpers
compatible with the rest of the pipeline.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

ModelType = Literal[
    "logistic_regression",
    "random_forest",
    "svm",
    "gradient_boosting",
    "lda",
]


class BaselineModel:
    """Scikit-learn baseline classifier for ECG beat classification.

    Parameters
    ----------
    model_type:
        Which classifier to use as the estimator.  One of
        ``"logistic_regression"`` (default), ``"random_forest"``,
        ``"svm"``, ``"gradient_boosting"``, or ``"lda"``.
    **kwargs:
        Extra keyword arguments forwarded to the chosen estimator.
    """

    _ESTIMATORS: Dict[str, Any] = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "svm": SVC,
        "gradient_boosting": GradientBoostingClassifier,
        "lda": LinearDiscriminantAnalysis,
    }

    def __init__(
        self,
        model_type: ModelType = "logistic_regression",
        **kwargs: Any,
    ) -> None:
        if model_type not in self._ESTIMATORS:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from {list(self._ESTIMATORS.keys())}."
            )
        self.model_type: ModelType = model_type
        self._kwargs: Dict[str, Any] = kwargs
        self.pipeline: Optional[Pipeline] = None
        self._feature_names: Optional[list] = None
        self._classes: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    #  Build                                                               #
    # ------------------------------------------------------------------ #

    def build_pipeline(self) -> Pipeline:
        """Construct and return the sklearn ``Pipeline``.

        The pipeline consists of:
        1. :class:`~sklearn.preprocessing.StandardScaler`
        2. The chosen classifier.

        Returns
        -------
        sklearn.pipeline.Pipeline
        """
        cls = self._ESTIMATORS[self.model_type]

        # Provide sensible defaults for each classifier
        defaults: Dict[str, Any] = {
            "logistic_regression": {
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": 42,
                "solver": "lbfgs",
                "multi_class": "multinomial",
                "C": 1.0,
            },
            "random_forest": {
                "n_estimators": 200,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            },
            "svm": {
                "probability": True,
                "class_weight": "balanced",
                "random_state": 42,
                "kernel": "rbf",
            },
            "gradient_boosting": {
                "n_estimators": 200,
                "random_state": 42,
                "learning_rate": 0.1,
            },
            "lda": {},
        }

        params = {**defaults[self.model_type], **self._kwargs}
        estimator = cls(**params)

        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", estimator),
            ]
        )
        logger.info("Built pipeline: StandardScaler + %s", cls.__name__)
        return self.pipeline

    # ------------------------------------------------------------------ #
    #  Train                                                               #
    # ------------------------------------------------------------------ #

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "BaselineModel":
        """Fit the pipeline on training data.

        Parameters
        ----------
        X_train:
            Feature matrix of shape ``(n_samples, n_features)``.
        y_train:
            Target integer labels of shape ``(n_samples,)``.

        Returns
        -------
        self
        """
        if self.pipeline is None:
            self.build_pipeline()
        logger.info(
            "Training %s on %d samples, %d features …",
            self.model_type,
            X_train.shape[0],
            X_train.shape[1],
        )
        self.pipeline.fit(X_train, y_train)  # type: ignore[union-attr]
        self._classes = self.pipeline.classes_  # type: ignore[union-attr]
        logger.info("Training complete.")
        return self

    # ------------------------------------------------------------------ #
    #  Predict                                                             #
    # ------------------------------------------------------------------ #

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions for *X*.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Predicted integer class labels, shape ``(n_samples,)``.
        """
        self._check_fitted()
        return self.pipeline.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities if the classifier supports them.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        np.ndarray of shape ``(n_samples, n_classes)``.
        """
        self._check_fitted()
        clf = self.pipeline["classifier"]  # type: ignore[index]
        if hasattr(clf, "predict_proba"):
            return self.pipeline.predict_proba(X)  # type: ignore[union-attr]
        raise AttributeError(
            f"Classifier '{type(clf).__name__}' does not support predict_proba."
        )

    # ------------------------------------------------------------------ #
    #  Evaluate                                                            #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_names: Optional[list] = None,
    ) -> str:
        """Evaluate on test data and return a classification report string.

        Parameters
        ----------
        X_test:
            Test feature matrix.
        y_test:
            True integer labels.
        target_names:
            Optional list of class name strings.

        Returns
        -------
        str
            Formatted :func:`~sklearn.metrics.classification_report`.
        """
        self._check_fitted()
        y_pred = self.predict(X_test)
        report = classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            zero_division=0,
        )
        logger.info("Evaluation report:\n%s", report)
        return report

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Serialize the fitted pipeline to *path* using pickle.

        Parameters
        ----------
        path:
            File path (should end in ``.pkl``).
        """
        self._check_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "model_type": self.model_type,
            "kwargs": self._kwargs,
            "pipeline": self.pipeline,
            "classes": self._classes,
            "feature_names": self._feature_names,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> "BaselineModel":
        """Load a previously saved model from *path*.

        Parameters
        ----------
        path:
            Path to the ``.pkl`` file created by :meth:`save`.

        Returns
        -------
        self (with pipeline restored)
        """
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        self.model_type = payload["model_type"]
        self._kwargs = payload["kwargs"]
        self.pipeline = payload["pipeline"]
        self._classes = payload["classes"]
        self._feature_names = payload["feature_names"]
        logger.info("Model loaded from %s", path)
        return self

    # ------------------------------------------------------------------ #
    #  Feature importance                                                  #
    # ------------------------------------------------------------------ #

    def get_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Compute permutation feature importance on the provided dataset.

        Parameters
        ----------
        X:
            Feature matrix (pre-scaled or raw – the pipeline handles scaling).
        y:
            True labels.
        n_repeats:
            Number of permutation repeats.
        random_state:
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance_mean``, ``importance_std``
            sorted by descending mean importance.
        """
        self._check_fitted()
        result = permutation_importance(
            self.pipeline,  # type: ignore[arg-type]
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )
        n_features = X.shape[1]
        names = (
            self._feature_names
            if self._feature_names and len(self._feature_names) == n_features
            else [f"feature_{i}" for i in range(n_features)]
        )
        df = pd.DataFrame(
            {
                "feature": names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _check_fitted(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted. Call train() first.")
