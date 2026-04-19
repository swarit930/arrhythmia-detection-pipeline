"""
train_baseline.py – Train a scikit-learn baseline arrhythmia classifier.

Usage
-----
    python scripts/train_baseline.py [--data-dir DATA_DIR] [--model-type logistic_regression]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import ECGDataLoader
from src.preprocessor import ECGPreprocessor
from src.feature_extractor import FeatureExtractor
from src.baseline_model import BaselineModel
from src.evaluator import PatientSplitter, ModelEvaluator
from src.evaluation_report import EvaluationReport
from src.utils import setup_logging, set_seed, ensure_dir, save_json

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline ECG classifier.")
    parser.add_argument("--data-dir", default=config.DATA_DIR, help="MIT-BIH data directory")
    parser.add_argument("--model-dir", default=config.MODEL_DIR, help="Output model directory")
    parser.add_argument("--results-dir", default=config.RESULTS_DIR, help="Results directory")
    parser.add_argument(
        "--model-type",
        default="random_forest",
        choices=["logistic_regression", "random_forest", "svm", "gradient_boosting", "lda"],
        help="Baseline classifier",
    )
    parser.add_argument(
        "--records",
        nargs="+",
        default=None,
        help="Subset of records to use (default: all 48)",
    )
    parser.add_argument("--download", action="store_true", help="Download records if missing")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_features_and_labels(
    records: list,
    data_dir: str,
    download: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all records, extract features, and map labels to AAMI indices."""
    all_features: list = []
    all_labels: list = []

    for rec_name, signal, fields, annotation in ECGDataLoader.load_all_records(records, data_dir):
        if annotation is None:
            logger.warning("Skipping %s: no annotation.", rec_name)
            continue

        fs = fields["fs"]
        clean = ECGPreprocessor.preprocess_record(signal, fs=fs)
        r_peaks = ECGPreprocessor.detect_r_peaks(clean, fs=fs)

        # Map annotation symbols to AAMI integer labels
        ann_samples = np.array(annotation.sample)
        ann_symbols = np.array(annotation.symbol)

        # Align annotation to our detected r_peaks (keep only annotated beats)
        labels_raw: list = []
        peaks_filtered: list = []
        for sym, samp in zip(ann_symbols, ann_samples):
            aami = config.MITBIH_TO_AAMI.get(sym)
            if aami is None:
                continue
            peaks_filtered.append(samp)
            labels_raw.append(config.AAMI_CLASS_TO_IDX[aami])

        if len(peaks_filtered) == 0:
            continue

        r_peaks_ann = np.array(peaks_filtered, dtype=np.int64)
        labels_arr = np.array(labels_raw, dtype=np.int64)

        beats, labels = ECGPreprocessor.extract_beats(clean, r_peaks_ann, labels_arr, fs=fs)
        if len(beats) == 0:
            continue

        feat_df = FeatureExtractor.extract_all_features(beats, r_peaks_ann[: len(beats)], fs=fs)
        all_features.append(feat_df)
        all_labels.append(labels[: len(feat_df)])

    if not all_features:
        raise RuntimeError("No features extracted. Check data_dir and records.")

    X = pd.concat(all_features, ignore_index=True).values.astype(np.float32)
    y = np.concatenate(all_labels).astype(np.int64)
    return X, y


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level, os.path.join(config.LOG_DIR, "train_baseline.log"))
    set_seed(args.seed)
    ensure_dir(args.model_dir)
    ensure_dir(args.results_dir)

    records = args.records or config.MITBIH_RECORDS

    if args.download:
        for rec in records:
            try:
                ECGDataLoader.download_record(rec, args.data_dir)
            except Exception as exc:
                logger.warning("Download failed for %s: %s", rec, exc)

    # Patient-aware split
    splitter = PatientSplitter(random_state=args.seed)
    train_recs, val_recs, test_recs = splitter.split(records)
    logger.info("Records – train: %d, val: %d, test: %d", len(train_recs), len(val_recs), len(test_recs))

    logger.info("Extracting training features …")
    X_train, y_train = load_features_and_labels(train_recs, args.data_dir)
    logger.info("Training set: %d beats, %d features.", *X_train.shape)

    logger.info("Extracting validation features …")
    X_val, y_val = load_features_and_labels(val_recs, args.data_dir)

    logger.info("Extracting test features …")
    X_test, y_test = load_features_and_labels(test_recs, args.data_dir)

    # Train
    model = BaselineModel(model_type=args.model_type)
    model.build_pipeline()
    model.train(X_train, y_train)

    # Evaluate on validation set
    val_report = model.evaluate(X_val, y_val, target_names=config.AAMI_CLASSES)
    logger.info("Validation report:\n%s", val_report)

    # Evaluate on test set
    report = EvaluationReport(class_names=config.AAMI_CLASSES)
    metrics = report.generate_baseline_report(model, X_test, y_test, save_dir=args.results_dir)
    logger.info("Test macro F1: %.4f", metrics.get("macro_f1", 0))

    # Save model
    model_path = os.path.join(args.model_dir, f"baseline_{args.model_type}.pkl")
    model.save(model_path)
    logger.info("Model saved to %s", model_path)

    # Feature importance
    try:
        fi = model.get_feature_importance(X_test, y_test, n_repeats=5)
        logger.info("Top-5 features:\n%s", fi.head(5).to_string(index=False))
        fi.to_csv(os.path.join(args.results_dir, "feature_importance.csv"), index=False)
    except Exception as exc:
        logger.warning("Feature importance failed: %s", exc)


if __name__ == "__main__":
    main()
