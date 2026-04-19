"""
Unit tests for PyTorch models, loss functions, and training utilities.

All tests are purely synthetic – no real ECG data or downloads are needed.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models_pytorch import ECG_CNN, ECG_LSTM
from src.loss_functions import FocalLoss, WeightedCrossEntropyLoss, compute_class_weights
from src.trainer import Trainer
from src.utils import AverageMeter, EarlyStopping

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 5
BEAT_LEN = 129
SEQ_LEN = 10
BATCH = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_cnn_batch(batch: int = BATCH) -> torch.Tensor:
    """Return a random (batch, 1, BEAT_LEN) CNN input tensor."""
    return torch.randn(batch, 1, BEAT_LEN)


def random_lstm_batch(batch: int = BATCH, seq: int = SEQ_LEN) -> torch.Tensor:
    """Return a random (batch, seq_len, BEAT_LEN) LSTM input tensor."""
    return torch.randn(batch, seq, BEAT_LEN)


def random_labels(batch: int = BATCH) -> torch.Tensor:
    return torch.randint(0, NUM_CLASSES, (batch,))


# ---------------------------------------------------------------------------
# test_cnn_forward_pass
# ---------------------------------------------------------------------------

class TestECGCNN:
    def test_output_shape(self):
        model = ECG_CNN(num_classes=NUM_CLASSES, input_length=BEAT_LEN)
        model.eval()
        x = random_cnn_batch()
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_output_dtype(self):
        model = ECG_CNN(num_classes=NUM_CLASSES)
        x = random_cnn_batch()
        with torch.no_grad():
            logits = model(x)
        assert logits.dtype == torch.float32

    def test_no_nan_output(self):
        model = ECG_CNN(num_classes=NUM_CLASSES)
        x = random_cnn_batch()
        with torch.no_grad():
            logits = model(x)
        assert not torch.isnan(logits).any()

    def test_single_sample(self):
        """Forward pass should work for batch size 1."""
        model = ECG_CNN(num_classes=NUM_CLASSES)
        x = random_cnn_batch(batch=1)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, NUM_CLASSES)

    def test_feature_maps_shape(self):
        """get_feature_maps should return (batch, 128) embedding."""
        model = ECG_CNN(num_classes=NUM_CLASSES)
        x = random_cnn_batch()
        with torch.no_grad():
            feats = model.get_feature_maps(x)
        assert feats.shape == (BATCH, 128)

    def test_gradient_flows(self):
        """Loss.backward() should produce non-None gradients on all parameters."""
        model = ECG_CNN(num_classes=NUM_CLASSES)
        x = random_cnn_batch()
        y = random_labels()
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# test_lstm_forward_pass
# ---------------------------------------------------------------------------

class TestECGLSTM:
    def test_output_shape(self):
        model = ECG_LSTM(num_classes=NUM_CLASSES, beat_len=BEAT_LEN, seq_len=SEQ_LEN)
        model.eval()
        x = random_lstm_batch()
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_no_nan_output(self):
        model = ECG_LSTM(num_classes=NUM_CLASSES, beat_len=BEAT_LEN)
        x = random_lstm_batch()
        with torch.no_grad():
            logits = model(x)
        assert not torch.isnan(logits).any()

    def test_return_attention(self):
        """return_attention=True should yield logits and weights."""
        model = ECG_LSTM(num_classes=NUM_CLASSES, beat_len=BEAT_LEN, seq_len=SEQ_LEN)
        model.eval()
        x = random_lstm_batch()
        with torch.no_grad():
            logits, weights = model(x, return_attention=True)
        assert logits.shape == (BATCH, NUM_CLASSES)
        assert weights.shape == (BATCH, SEQ_LEN)

    def test_attention_sums_to_one(self):
        """Attention weights (softmax output) should sum to ~1 per sample."""
        model = ECG_LSTM(num_classes=NUM_CLASSES, beat_len=BEAT_LEN, seq_len=SEQ_LEN)
        model.eval()
        x = random_lstm_batch()
        with torch.no_grad():
            _, weights = model(x, return_attention=True)
        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-4)

    def test_get_attention_weights(self):
        model = ECG_LSTM(num_classes=NUM_CLASSES, beat_len=BEAT_LEN, seq_len=SEQ_LEN)
        model.eval()
        x = random_lstm_batch()
        with torch.no_grad():
            w = model.get_attention_weights(x)
        assert w.shape == (BATCH, SEQ_LEN)


# ---------------------------------------------------------------------------
# test_focal_loss
# ---------------------------------------------------------------------------

class TestFocalLoss:
    def test_scalar_output(self):
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(16, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (16,))
        loss = loss_fn(logits, targets)
        assert loss.shape == ()

    def test_non_negative(self):
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(32, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (32,))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0

    def test_per_sample_reduction_none(self):
        loss_fn = FocalLoss(gamma=2.0, reduction="none")
        logits = torch.randn(8, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (8,))
        loss = loss_fn(logits, targets)
        assert loss.shape == (8,)

    def test_perfect_prediction_low_loss(self):
        """When logits strongly predict the correct class, loss should be near 0."""
        loss_fn = FocalLoss(gamma=2.0)
        # One-hot-like: push the correct class to a very high logit
        targets = torch.zeros(4, dtype=torch.long)  # all class 0
        logits = torch.full((4, NUM_CLASSES), -10.0)
        logits[:, 0] = 10.0  # confident correct prediction
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.01

    def test_weighted_alpha(self):
        """Per-class alpha weights should not cause NaN."""
        alpha = torch.ones(NUM_CLASSES)
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        logits = torch.randn(16, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (16,))
        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# test_weighted_ce_loss
# ---------------------------------------------------------------------------

class TestWeightedCELoss:
    def test_scalar_output(self):
        weights = torch.ones(NUM_CLASSES)
        loss_fn = WeightedCrossEntropyLoss(weights)
        logits = torch.randn(16, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (16,))
        loss = loss_fn(logits, targets)
        assert loss.shape == ()

    def test_matches_standard_ce_with_uniform_weights(self):
        """Uniform weights should give the same result as standard cross-entropy."""
        weights = torch.ones(NUM_CLASSES)
        weighted_loss_fn = WeightedCrossEntropyLoss(weights)
        std_loss_fn = nn.CrossEntropyLoss(weight=weights)
        logits = torch.randn(32, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (32,))
        wl = weighted_loss_fn(logits, targets)
        sl = std_loss_fn(logits, targets)
        assert torch.allclose(wl, sl, atol=1e-5)

    def test_non_negative(self):
        weights = compute_class_weights(np.array([0, 0, 1, 2, 3, 4] * 10))
        loss_fn = WeightedCrossEntropyLoss(weights)
        logits = torch.randn(16, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (16,))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# test_compute_class_weights
# ---------------------------------------------------------------------------

class TestComputeClassWeights:
    def test_output_shape(self):
        labels = np.array([0, 1, 2, 3, 4] * 10)
        w = compute_class_weights(labels, num_classes=5)
        assert w.shape == (5,)

    def test_mean_one(self):
        """Weights should be normalised so their mean equals 1."""
        labels = np.array([0] * 100 + [1] * 10 + [2] * 5 + [3] * 3 + [4] * 2)
        w = compute_class_weights(labels, num_classes=5)
        assert abs(w.mean().item() - 1.0) < 1e-4

    def test_rare_class_higher_weight(self):
        """A rare class should receive a higher weight than a common class."""
        labels = np.array([0] * 900 + [1] * 100)
        w = compute_class_weights(labels, num_classes=2)
        assert w[1].item() > w[0].item()

    def test_accepts_tensor(self):
        labels = torch.tensor([0, 1, 2, 3, 4] * 5)
        w = compute_class_weights(labels, num_classes=5)
        assert isinstance(w, torch.Tensor)


# ---------------------------------------------------------------------------
# test_early_stopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_no_stop_on_improvement(self):
        es = EarlyStopping(patience=3)
        for loss in [1.0, 0.9, 0.8, 0.7]:
            es(loss)
        assert not es.should_stop

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3)
        es(1.0)  # improvement
        es(1.1)
        es(1.2)
        es(1.3)  # counter = 3 → stop
        assert es.should_stop

    def test_counter_resets_on_improvement(self):
        es = EarlyStopping(patience=3)
        es(1.0)
        es(1.1)   # counter = 1
        es(1.2)   # counter = 2
        es(0.5)   # improvement → counter resets to 0
        assert es.counter == 0
        assert not es.should_stop

    def test_reset(self):
        es = EarlyStopping(patience=2)
        es(1.0)
        es(1.1)
        es(1.2)
        assert es.should_stop
        es.reset()
        assert not es.should_stop
        assert es.counter == 0


# ---------------------------------------------------------------------------
# test_average_meter
# ---------------------------------------------------------------------------

class TestAverageMeter:
    def test_basic_average(self):
        meter = AverageMeter()
        meter.update(1.0, n=1)
        meter.update(3.0, n=1)
        assert meter.avg == pytest.approx(2.0)

    def test_weighted_average(self):
        meter = AverageMeter()
        meter.update(1.0, n=10)
        meter.update(5.0, n=10)
        assert meter.avg == pytest.approx(3.0)

    def test_reset(self):
        meter = AverageMeter()
        meter.update(5.0, n=1)
        meter.reset()
        assert meter.avg == 0.0
        assert meter.count == 0


# ---------------------------------------------------------------------------
# test_trainer_step
# ---------------------------------------------------------------------------

class TestTrainerStep:
    def _make_dummy_loader(self, n_batches: int = 4, batch_size: int = 8):
        X = torch.randn(n_batches * batch_size, 1, BEAT_LEN)
        y = torch.randint(0, NUM_CLASSES, (n_batches * batch_size,))
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def test_train_epoch_returns_loss_acc(self):
        model = ECG_CNN(num_classes=NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, device="cpu")
        loader = self._make_dummy_loader()
        loss, acc = trainer.train_epoch(loader)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_validate_epoch_returns_loss_acc(self):
        model = ECG_CNN(num_classes=NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, device="cpu")
        loader = self._make_dummy_loader()
        loss, acc = trainer.validate_epoch(loader)
        assert isinstance(loss, float)
        assert 0.0 <= acc <= 1.0

    def test_loss_decreases_over_epochs(self):
        """Loss should trend down on a tiny dataset that can be memorised."""
        torch.manual_seed(0)
        # Very small dataset (one batch)
        X = torch.randn(16, 1, BEAT_LEN)
        y = torch.zeros(16, dtype=torch.long)  # all same class → easy to memorise
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=16)

        model = ECG_CNN(num_classes=NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, device="cpu")

        losses = []
        for _ in range(5):
            loss, _ = trainer.train_epoch(loader)
            losses.append(loss)

        # First loss should be higher than last
        assert losses[0] > losses[-1]

    def test_full_train_loop(self):
        """train() should run without error for 2 epochs and return history."""
        torch.manual_seed(1)
        X = torch.randn(32, 1, BEAT_LEN)
        y = torch.randint(0, NUM_CLASSES, (32,))
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=16)

        model = ECG_CNN(num_classes=NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(
            model, optimizer, criterion, device="cpu",
            config={"early_stopping_patience": 10},
        )
        history = trainer.train(loader, loader, epochs=2, save_path=None)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 2
