"""
Training loop for PyTorch ECG classification models.

:class:`Trainer` handles:
* Per-epoch training and validation passes
* Learning-rate scheduling (ReduceLROnPlateau)
* Early stopping (patience-based)
* Best-model checkpointing
* Per-epoch metric logging
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.utils import AverageMeter, EarlyStopping

logger = logging.getLogger(__name__)


class Trainer:
    """Generic PyTorch trainer with early stopping and LR scheduling.

    Parameters
    ----------
    model:
        PyTorch model to train.
    optimizer:
        Optimiser instance (e.g. ``torch.optim.Adam``).
    criterion:
        Loss function callable that accepts ``(logits, targets)``.
    device:
        ``torch.device`` or string (e.g. ``"cuda"`` or ``"cpu"``).
    config:
        Optional dict with hyper-parameters; recognised keys:
        ``lr_scheduler_patience``, ``lr_scheduler_factor``,
        ``early_stopping_patience``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device | str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device) if isinstance(device, str) else device
        self.config = config or {}

        self.model.to(self.device)

        # LR scheduler
        lr_patience = int(self.config.get("lr_scheduler_patience", 5))
        lr_factor = float(self.config.get("lr_scheduler_factor", 0.5))
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=lr_patience,
            factor=lr_factor,
        )

        # History
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

    # ------------------------------------------------------------------ #
    #  Single-epoch passes                                                 #
    # ------------------------------------------------------------------ #

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Run one training epoch.

        Parameters
        ----------
        dataloader:
            Training :class:`~torch.utils.data.DataLoader`.

        Returns
        -------
        avg_loss : float
        accuracy : float  (fraction, 0–1)
        """
        self.model.train()
        loss_meter = AverageMeter()
        correct = 0
        total = 0

        for batch in dataloader:
            inputs, targets = self._unpack_batch(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            batch_size = targets.size(0)
            loss_meter.update(loss.item(), batch_size)

            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += batch_size

        avg_loss = loss_meter.avg
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Run one validation epoch (no gradient computation).

        Parameters
        ----------
        dataloader:
            Validation :class:`~torch.utils.data.DataLoader`.

        Returns
        -------
        avg_loss : float
        accuracy : float
        """
        self.model.eval()
        loss_meter = AverageMeter()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = self._unpack_batch(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

                batch_size = targets.size(0)
                loss_meter.update(loss.item(), batch_size)

                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += batch_size

        avg_loss = loss_meter.avg
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    # ------------------------------------------------------------------ #
    #  Full training loop                                                  #
    # ------------------------------------------------------------------ #

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        save_path: Optional[str] = None,
    ) -> Dict[str, list]:
        """Train the model for up to *epochs* epochs with early stopping.

        Parameters
        ----------
        train_loader:
            DataLoader for training data.
        val_loader:
            DataLoader for validation data.
        epochs:
            Maximum number of epochs.
        save_path:
            File path to save the best model checkpoint
            (e.g. ``"models/cnn_best.pth"``).  If ``None``, no checkpoint
            is saved to disk.

        Returns
        -------
        dict
            Training history with keys ``train_loss``, ``val_loss``,
            ``train_acc``, ``val_acc``, ``lr``.
        """
        patience = int(self.config.get("early_stopping_patience", 10))
        early_stopper = EarlyStopping(patience=patience, verbose=True)

        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            elapsed = time.time() - t0

            # LR scheduling
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            logger.info(
                "Epoch %3d/%d | train_loss=%.4f acc=%.4f | "
                "val_loss=%.4f acc=%.4f | lr=%.2e | %.1fs",
                epoch, epochs,
                train_loss, train_acc,
                val_loss, val_acc,
                current_lr, elapsed,
            )

            # Checkpoint best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path is not None:
                    self._save_checkpoint(save_path, epoch, val_loss, val_acc)

            # Early stopping
            early_stopper(val_loss)
            if early_stopper.should_stop:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        # Restore best weights from checkpoint
        if save_path is not None and os.path.exists(save_path):
            self.load_checkpoint(save_path)
            logger.info("Restored best model from %s.", save_path)

        return self.history

    # ------------------------------------------------------------------ #
    #  Checkpointing                                                       #
    # ------------------------------------------------------------------ #

    def _save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_loss: float,
        val_acc: float,
    ) -> None:
        """Save model weights and optimizer state to *path*."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        torch.save(checkpoint, path)
        logger.debug("Checkpoint saved → %s (epoch %d, val_loss=%.4f)", path, epoch, val_loss)

    def load_checkpoint(self, path: str) -> None:
        """Load model (and optionally optimiser) weights from a checkpoint.

        Parameters
        ----------
        path:
            Path to the ``.pth`` checkpoint file created by
            :meth:`_save_checkpoint`.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                logger.warning("Could not restore optimizer state from checkpoint.")
        logger.info(
            "Loaded checkpoint from %s (epoch %d, val_loss=%.4f).",
            path,
            checkpoint.get("epoch", -1),
            checkpoint.get("val_loss", float("nan")),
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _unpack_batch(
        batch: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unpack a batch that is either a (inputs, targets) tuple or a dict."""
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch[0], batch[1]
        elif isinstance(batch, dict):
            inputs = batch["input"]
            targets = batch["label"]
        else:
            raise TypeError(f"Unexpected batch type: {type(batch)}")
        return inputs, targets.long()
