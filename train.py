from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, TypedDict

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import config
from dataset import ArrhythmiaDataset, SignalAugment, build_weighted_sampler, collect_beats_with_rr_features
from model import MultiModalECGNet
from src.evaluator import PatientSplitter
from src.loss_functions import FocalLoss
from src.utils import ensure_dir, set_seed, setup_logging

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    loss: float
    acc: float
    macro_f1: float


class TestMetrics(TypedDict):
    macro_f1: float
    weighted_f1: float
    classification_report: Dict[str, object]
    confusion_matrix: list[list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal ECG arrhythmia model.")
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR)
    parser.add_argument("--model-dir", type=str, default=config.MODEL_DIR)
    parser.add_argument("--results-dir", type=str, default=config.RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience on val macro-F1.")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--records", nargs="+", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def verify_patient_wise_split(train_records, val_records, test_records) -> None:
    train_ids = set([r[:3] for r in train_records])
    val_ids = set([r[:3] for r in val_records])
    test_ids = set([r[:3] for r in test_records])
    assert train_ids.isdisjoint(val_ids), "Patient leakage detected between train and val sets."
    assert train_ids.isdisjoint(test_ids), "Patient leakage detected between train and test sets."
    assert val_ids.isdisjoint(test_ids), "Patient leakage detected between val and test sets."


def compute_inverse_frequency_weights(labels: np.ndarray, num_classes: int = config.NUM_CLASSES) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts[counts == 0.0] = 1.0
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: MultiModalECGNet,
    loader: DataLoader,
    criterion: FocalLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total = 0
    correct = 0
    all_true = []
    all_pred = []

    for signal, rr, target in loader:
        signal = signal.to(device)
        rr = rr.to(device)
        target = target.to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(signal, rr)
            loss = criterion(logits, target)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        bs = target.size(0)
        total_loss += loss.item() * bs
        total += bs
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        all_true.append(target.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if len(y_true) else 0.0
    return EpochMetrics(
        loss=total_loss / max(total, 1),
        acc=correct / max(total, 1),
        macro_f1=float(macro_f1),
    )


def evaluate_test_set(model: MultiModalECGNet, loader: DataLoader, device: torch.device) -> TestMetrics:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for signal, rr, target in loader:
            signal = signal.to(device)
            rr = rr.to(device)
            logits = model(signal, rr)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_pred.append(pred)
            y_true.append(target.numpy())

    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)
    report = classification_report(
        y_true_np,
        y_pred_np,
        target_names=config.AAMI_CLASSES,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true_np, y_pred_np)
    macro_f1 = f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level, os.path.join(config.LOG_DIR, "train_multimodal.log"))
    set_seed(args.seed)
    ensure_dir(args.model_dir)
    ensure_dir(args.results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    records = args.records or config.MITBIH_RECORDS

    if args.download:
        from src.data_loader import ECGDataLoader

        for rec in records:
            ECGDataLoader.download_record(rec, args.data_dir)

    splitter = PatientSplitter(random_state=args.seed)
    train_records, val_records, test_records = splitter.split(records, test_size=config.TEST_SIZE, val_size=config.VAL_SIZE)
    verify_patient_wise_split(train_records, val_records, test_records)

    logger.info("Preparing training set...")
    train_data = collect_beats_with_rr_features(train_records, args.data_dir)
    logger.info("Preparing validation set...")
    val_data = collect_beats_with_rr_features(val_records, args.data_dir)
    logger.info("Preparing test set...")
    test_data = collect_beats_with_rr_features(test_records, args.data_dir)

    augment = SignalAugment()
    train_dataset = ArrhythmiaDataset(
        beats=train_data.beats,
        rr_features=train_data.rr_features,
        labels=train_data.labels,
        record_ids=train_data.record_ids,
        augment=augment,
    )
    val_dataset = ArrhythmiaDataset(val_data.beats, val_data.rr_features, val_data.labels, record_ids=val_data.record_ids)
    test_dataset = ArrhythmiaDataset(test_data.beats, test_data.rr_features, test_data.labels, record_ids=test_data.record_ids)

    train_sampler = build_weighted_sampler(train_data.labels, num_classes=config.NUM_CLASSES)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = MultiModalECGNet(num_classes=config.NUM_CLASSES, dropout=config.DROPOUT_RATE).to(device)
    class_weights = compute_inverse_frequency_weights(train_data.labels, num_classes=config.NUM_CLASSES).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_macro_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    history = []
    best_ckpt_path = os.path.join(args.model_dir, "multimodal_best.pth")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_metrics.macro_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f train_acc=%.4f train_macro_f1=%.4f | "
            "val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f | lr=%.2e",
            epoch,
            args.epochs,
            train_metrics.loss,
            train_metrics.acc,
            train_metrics.macro_f1,
            val_metrics.loss,
            val_metrics.acc,
            val_metrics.macro_f1,
            current_lr,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_acc": train_metrics.acc,
                "train_macro_f1": train_metrics.macro_f1,
                "val_loss": val_metrics.loss,
                "val_acc": val_metrics.acc,
                "val_macro_f1": val_metrics.macro_f1,
                "lr": current_lr,
            }
        )

        if val_metrics.macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_metrics.macro_f1
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1": best_val_macro_f1,
                },
                best_ckpt_path,
            )
        else:
            no_improve += 1

        if no_improve >= args.patience:
            logger.info("Early stopping on val_macro_f1 at epoch %d.", epoch)
            break

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded best checkpoint from epoch %d with val_macro_f1=%.4f", best_epoch, best_val_macro_f1)

    test_metrics = evaluate_test_set(model, test_loader, device)
    logger.info("Test macro_f1=%.4f | weighted_f1=%.4f", test_metrics["macro_f1"], test_metrics["weighted_f1"])

    np.save(os.path.join(args.results_dir, "multimodal_history.npy"), np.array(history, dtype=object), allow_pickle=True)
    with open(os.path.join(args.results_dir, "multimodal_test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best val macro F1: {best_val_macro_f1:.6f}\n")
        f.write(f"Test macro F1: {test_metrics['macro_f1']:.6f}\n")
        f.write(f"Test weighted F1: {test_metrics['weighted_f1']:.6f}\n")
        f.write("Confusion matrix:\n")
        f.write(str(test_metrics["confusion_matrix"]))
        f.write("\n\nClassification report:\n")
        f.write(str(test_metrics["classification_report"]))


if __name__ == "__main__":
    main()
