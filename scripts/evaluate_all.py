"""
evaluate_all.py – Load all trained models and generate a comparison report.

Usage
-----
    python scripts/evaluate_all.py [--data-dir DATA_DIR]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import ECGDataLoader
from src.preprocessor import ECGPreprocessor
from src.feature_extractor import FeatureExtractor
from src.baseline_model import BaselineModel
from src.models_pytorch import ECG_CNN, ECG_LSTM
from src.evaluator import PatientSplitter, ModelEvaluator
from src.evaluation_report import EvaluationReport
from src.utils import setup_logging, set_seed, ensure_dir, load_json

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all trained models.")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--model-dir", default=config.MODEL_DIR)
    parser.add_argument("--results-dir", default=config.RESULTS_DIR)
    parser.add_argument("--records", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def collect_beats_and_labels(records, data_dir):
    beats_list, labels_list = [], []
    for _, signal, fields, annotation in ECGDataLoader.load_all_records(records, data_dir):
        if annotation is None:
            continue
        fs = fields["fs"]
        clean = ECGPreprocessor.preprocess_record(signal, fs=fs)
        peaks, labels_raw = [], []
        for sym, samp in zip(annotation.symbol, annotation.sample):
            aami = config.MITBIH_TO_AAMI.get(sym)
            if aami is None:
                continue
            peaks.append(samp)
            labels_raw.append(config.AAMI_CLASS_TO_IDX[aami])
        if not peaks:
            continue
        r_peaks = np.array(peaks, dtype=np.int64)
        labels = np.array(labels_raw, dtype=np.int64)
        beats, valid_labels = ECGPreprocessor.extract_beats(clean, r_peaks, labels, fs=fs)
        if len(beats):
            beats_list.append(beats)
            labels_list.append(valid_labels)
    if not beats_list:
        raise RuntimeError("No beats extracted.")
    return (
        np.concatenate(beats_list).astype(np.float32),
        np.concatenate(labels_list).astype(np.int64),
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level, os.path.join(config.LOG_DIR, "evaluate_all.log"))
    set_seed(args.seed)
    ensure_dir(args.results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = args.records or config.MITBIH_RECORDS

    # Use the same patient split
    splitter = PatientSplitter(random_state=args.seed)
    _, _, test_recs = splitter.split(records)
    logger.info("Test records: %d", len(test_recs))

    X_test, y_test = collect_beats_and_labels(test_recs, args.data_dir)
    logger.info("Test beats: %d", len(X_test))

    evaluator = ModelEvaluator(class_names=config.AAMI_CLASSES)
    reporter = EvaluationReport(class_names=config.AAMI_CLASSES)
    results: dict = {}

    # ------------------------------------------------------------------
    # 1. Baseline models
    # ------------------------------------------------------------------
    # Extract features for baseline
    feat_df = FeatureExtractor.extract_all_features(
        X_test,
        np.arange(len(X_test), dtype=np.int64),
        fs=config.SAMPLING_RATE,
    )
    X_feat = feat_df.values.astype(np.float32)

    for model_type in ["logistic_regression", "random_forest"]:
        model_path = os.path.join(args.model_dir, f"baseline_{model_type}.pkl")
        if not os.path.exists(model_path):
            logger.warning("Baseline model not found: %s – skipping.", model_path)
            continue
        model = BaselineModel(model_type=model_type)
        model.load(model_path)
        save_dir = os.path.join(args.results_dir, model_type)
        metrics = reporter.generate_baseline_report(model, X_feat, y_test, save_dir=save_dir)
        results[model_type] = metrics
        logger.info("%s macro F1: %.4f", model_type, metrics.get("macro_f1", 0))

    # ------------------------------------------------------------------
    # 2. CNN
    # ------------------------------------------------------------------
    cnn_path = os.path.join(args.model_dir, "cnn_best.pth")
    if os.path.exists(cnn_path):
        cnn_model = ECG_CNN(num_classes=config.NUM_CLASSES)
        ckpt = torch.load(cnn_path, map_location=device)
        cnn_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        cnn_model.eval()

        X_t = torch.tensor(X_test).unsqueeze(1)
        y_t = torch.tensor(y_test)
        cnn_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=512, shuffle=False)
        save_dir = os.path.join(args.results_dir, "cnn")
        metrics = reporter.generate_dl_report(cnn_model, cnn_loader, device, save_dir=save_dir, model_name="cnn")
        results["cnn"] = metrics
        logger.info("CNN macro F1: %.4f", metrics.get("macro_f1", 0))
    else:
        logger.warning("CNN checkpoint not found: %s – skipping.", cnn_path)

    # ------------------------------------------------------------------
    # 3. LSTM
    # ------------------------------------------------------------------
    lstm_path = os.path.join(args.model_dir, "lstm_best.pth")
    if os.path.exists(lstm_path):
        from scripts.train_lstm import BeatSequenceDataset

        lstm_model = ECG_LSTM(num_classes=config.NUM_CLASSES)
        ckpt = torch.load(lstm_path, map_location=device)
        lstm_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        lstm_model.eval()

        lstm_ds = BeatSequenceDataset(X_test, y_test, seq_len=config.LSTM_SEQ_LEN)
        lstm_loader = DataLoader(lstm_ds, batch_size=512, shuffle=False)
        save_dir = os.path.join(args.results_dir, "lstm")
        metrics = reporter.generate_dl_report(lstm_model, lstm_loader, device, save_dir=save_dir, model_name="lstm")
        results["lstm"] = metrics
        logger.info("LSTM macro F1: %.4f", metrics.get("macro_f1", 0))
    else:
        logger.warning("LSTM checkpoint not found: %s – skipping.", lstm_path)

    # ------------------------------------------------------------------
    # 4. Comparison report
    # ------------------------------------------------------------------
    if results:
        reporter.generate_comparison_report(results, save_dir=args.results_dir)
        logger.info("Comparison report saved to %s.", args.results_dir)
    else:
        logger.warning("No models evaluated – comparison report skipped.")


if __name__ == "__main__":
    main()
