"""
train_lstm.py – Train the ECG_LSTM model on sequences of MIT-BIH beat segments.

The LSTM receives sequences of ``seq_len`` consecutive beats; the centre beat
is used as the classification target.

Usage
-----
    python scripts/train_lstm.py [--data-dir DATA_DIR] [--epochs 100]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import ECGDataLoader
from src.preprocessor import ECGPreprocessor
from src.models_pytorch import ECG_LSTM
from src.loss_functions import FocalLoss, compute_class_weights
from src.trainer import Trainer
from src.evaluator import PatientSplitter
from src.evaluation_report import EvaluationReport
from src.utils import setup_logging, set_seed, ensure_dir

logger = logging.getLogger(__name__)


# ======================================================================= #
#  Beat-sequence Dataset                                                   #
# ======================================================================= #

class BeatSequenceDataset(Dataset):
    """Dataset that yields sliding windows of consecutive beats.

    Each sample is a ``(seq_len, beat_len)`` tensor; the target is the label
    of the *centre* beat (index ``seq_len // 2``).

    Parameters
    ----------
    beats : np.ndarray, shape (N, beat_len)
    labels : np.ndarray, shape (N,)
    seq_len : int
    """

    def __init__(
        self,
        beats: np.ndarray,
        labels: np.ndarray,
        seq_len: int = config.LSTM_SEQ_LEN,
    ) -> None:
        self.beats = beats.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.seq_len = seq_len
        self.half = seq_len // 2
        # Valid indices: only those with enough context on both sides
        self.indices = list(range(self.half, len(beats) - (seq_len - self.half - 1)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        centre = self.indices[idx]
        start = centre - self.half
        end = start + self.seq_len
        seq = self.beats[start:end]          # (seq_len, beat_len)
        label = self.labels[centre]
        return torch.tensor(seq), torch.tensor(label)


# ======================================================================= #
#  Data loading helpers                                                    #
# ======================================================================= #

def collect_beats_and_labels(
    records: list,
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    beats_list: list = []
    labels_list: list = []

    for rec_name, signal, fields, annotation in ECGDataLoader.load_all_records(records, data_dir):
        if annotation is None:
            continue
        fs = fields["fs"]
        clean = ECGPreprocessor.preprocess_record(signal, fs=fs)

        peaks_filtered: list = []
        labels_raw: list = []
        for sym, samp in zip(annotation.symbol, annotation.sample):
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

    return (
        np.concatenate(beats_list).astype(np.float32),
        np.concatenate(labels_list).astype(np.int64),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ECG LSTM model.")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--model-dir", default=config.MODEL_DIR)
    parser.add_argument("--results-dir", default=config.RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--seq-len", type=int, default=config.LSTM_SEQ_LEN)
    parser.add_argument("--records", nargs="+", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level, os.path.join(config.LOG_DIR, "train_lstm.log"))
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

    train_ds = BeatSequenceDataset(X_train, y_train, seq_len=args.seq_len)
    val_ds = BeatSequenceDataset(X_val, y_val, seq_len=args.seq_len)
    test_ds = BeatSequenceDataset(X_test, y_test, seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ECG_LSTM(
        num_classes=config.NUM_CLASSES,
        beat_len=config.BEAT_WINDOW_SAMPLES,
        seq_len=args.seq_len,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.LSTM_NUM_LAYERS,
    )

    class_weights = compute_class_weights(y_train, num_classes=config.NUM_CLASSES)
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=config.FOCAL_LOSS_GAMMA)

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

    save_path = os.path.join(args.model_dir, "lstm_best.pth")
    logger.info("Starting LSTM training …")
    trainer.train(train_loader, val_loader, epochs=args.epochs, save_path=save_path)

    reporter = EvaluationReport(class_names=config.AAMI_CLASSES)
    metrics = reporter.generate_dl_report(
        trainer.model, test_loader, device,
        save_dir=args.results_dir,
        model_name="lstm",
    )
    logger.info("LSTM test macro F1: %.4f", metrics.get("macro_f1", 0))
    logger.info("Training complete. Best model saved to %s", save_path)


if __name__ == "__main__":
    main()
