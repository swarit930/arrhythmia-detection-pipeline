"""
train_cnn.py – Train the ECG_CNN model on MIT-BIH beat segments.

Usage
-----
    python scripts/train_cnn.py [--data-dir DATA_DIR] [--epochs 100]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import ECGDataLoader
from src.preprocessor import ECGPreprocessor
from src.models_pytorch import ECG_CNN
from src.loss_functions import FocalLoss, compute_class_weights
from src.trainer import Trainer
from src.evaluator import PatientSplitter, ModelEvaluator
from src.evaluation_report import EvaluationReport
from src.utils import setup_logging, set_seed, ensure_dir

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ECG CNN model.")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--model-dir", default=config.MODEL_DIR)
    parser.add_argument("--results-dir", default=config.RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--records", nargs="+", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def collect_beats_and_labels(
    records: list,
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (beats, labels) arrays from all records in *records*."""
    beats_list: list = []
    labels_list: list = []

    for rec_name, signal, fields, annotation in ECGDataLoader.load_all_records(records, data_dir):
        if annotation is None:
            continue
        fs = fields["fs"]
        clean = ECGPreprocessor.preprocess_record(signal, fs=fs)

        ann_samples = np.array(annotation.sample)
        ann_symbols = np.array(annotation.symbol)

        peaks_filtered: list = []
        labels_raw: list = []
        for sym, samp in zip(ann_symbols, ann_samples):
            aami = config.MITBIH_TO_AAMI.get(sym)
            if aami is None:
                continue
            peaks_filtered.append(samp)
            labels_raw.append(config.AAMI_CLASS_TO_IDX[aami])

        if not peaks_filtered:
            continue

        r_peaks = np.array(peaks_filtered, dtype=np.int64)
        labels = np.array(labels_raw, dtype=np.int64)
        beats, valid_labels = ECGPreprocessor.extract_beats(clean, r_peaks, labels, fs=fs)
        if len(beats) == 0:
            continue

        beats_list.append(beats)
        labels_list.append(valid_labels)

    if not beats_list:
        raise RuntimeError("No beats extracted.")

    X = np.concatenate(beats_list).astype(np.float32)
    y = np.concatenate(labels_list).astype(np.int64)
    return X, y


def make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap numpy arrays in a TensorDataset DataLoader.

    CNN expects input shape ``(batch, 1, 129)`` so we unsqueeze channel dim.
    """
    X_t = torch.tensor(X).unsqueeze(1)   # (N, 1, 129)
    y_t = torch.tensor(y)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level, os.path.join(config.LOG_DIR, "train_cnn.log"))
    set_seed(args.seed)
    ensure_dir(args.model_dir)
    ensure_dir(args.results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    records = args.records or config.MITBIH_RECORDS

    if args.download:
        for rec in records:
            try:
                ECGDataLoader.download_record(rec, args.data_dir)
            except Exception as exc:
                logger.warning("Download failed for %s: %s", rec, exc)

    splitter = PatientSplitter(random_state=args.seed)
    train_recs, val_recs, test_recs = splitter.split(records)

    logger.info("Loading training data …")
    X_train, y_train = collect_beats_and_labels(train_recs, args.data_dir)
    logger.info("Loading validation data …")
    X_val, y_val = collect_beats_and_labels(val_recs, args.data_dir)
    logger.info("Loading test data …")
    X_test, y_test = collect_beats_and_labels(test_recs, args.data_dir)

    train_loader = make_dataloader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_dataloader(X_test, y_test, args.batch_size, shuffle=False)

    # Model
    model = ECG_CNN(num_classes=config.NUM_CLASSES, input_length=config.BEAT_WINDOW_SAMPLES)

    # Loss: FocalLoss with inverse-frequency alpha weights
    class_weights = compute_class_weights(y_train, num_classes=config.NUM_CLASSES)
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=config.FOCAL_LOSS_GAMMA)

    # Optimiser
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )

    trainer_cfg = {
        "lr_scheduler_patience": config.LR_SCHEDULER_PATIENCE,
        "lr_scheduler_factor": config.LR_SCHEDULER_FACTOR,
        "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
    }
    trainer = Trainer(model, optimizer, criterion, device, config=trainer_cfg)

    save_path = os.path.join(args.model_dir, "cnn_best.pth")
    logger.info("Starting CNN training …")
    history = trainer.train(train_loader, val_loader, epochs=args.epochs, save_path=save_path)

    # Evaluation
    reporter = EvaluationReport(class_names=config.AAMI_CLASSES)
    metrics = reporter.generate_dl_report(
        trainer.model, test_loader, device,
        save_dir=args.results_dir,
        model_name="cnn",
    )
    logger.info("CNN test macro F1: %.4f", metrics.get("macro_f1", 0))
    logger.info("Training complete. Best model saved to %s", save_path)


if __name__ == "__main__":
    main()
