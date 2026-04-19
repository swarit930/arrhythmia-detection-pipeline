"""
PyTorch deep learning models for ECG arrhythmia classification.

Provides:
* :class:`ECG_CNN` – 1-D convolutional network for single-beat classification.
* :class:`ECG_LSTM` – Bidirectional LSTM with attention for sequence-level
  classification (multiple consecutive beats).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ======================================================================= #
#  ECG_CNN                                                                 #
# ======================================================================= #

class ECG_CNN(nn.Module):
    """1-D Convolutional Neural Network for single-beat ECG classification.

    Architecture
    ------------
    Three convolutional blocks:
      Conv1D(filters) → BatchNorm1d → ReLU → MaxPool1d(2)
    Filters: 32, 64, 128 | kernel_size=5 | pool_size=2

    Global Average Pooling → Dropout(0.5) → FC(128→64) → ReLU → FC(64→num_classes)

    Parameters
    ----------
    num_classes:
        Number of output classes (default 5 for AAMI: N, S, V, F, Q).
    input_length:
        Number of samples per beat (default 129).
    dropout:
        Dropout probability before the first FC layer.
    """

    def __init__(
        self,
        num_classes: int = 5,
        input_length: int = 129,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_length = input_length

        # Conv block 1: 1 → 32
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Conv block 2: 32 → 64
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Conv block 3: 64 → 128
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Global average pooling (produces a 128-dim vector)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, 1, input_length)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_classes)``.
        """
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Global Average Pool → (batch, 128, 1) → (batch, 128)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 128-dim embedding before the classifier head.

        Useful for visualisation and transfer learning.
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.global_avg_pool(x).squeeze(-1)
        return x


# ======================================================================= #
#  Attention mechanism helper                                              #
# ======================================================================= #

class _BeatAttention(nn.Module):
    """Additive (Bahdanau-style) attention over a sequence of beat encodings.

    Parameters
    ----------
    hidden_size:
        Dimensionality of the encoder hidden states.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(
        self, encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute context vector and attention weights.

        Parameters
        ----------
        encoder_outputs : torch.Tensor
            Shape ``(batch, seq_len, hidden_size)``.

        Returns
        -------
        context : torch.Tensor, shape ``(batch, hidden_size)``
        weights : torch.Tensor, shape ``(batch, seq_len)``
        """
        # scores: (batch, seq_len, 1) → (batch, seq_len)
        scores = self.attn(encoder_outputs).squeeze(-1)
        weights = F.softmax(scores, dim=-1)                    # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, hidden_size)
        return context, weights


# ======================================================================= #
#  ECG_LSTM                                                                #
# ======================================================================= #

class ECG_LSTM(nn.Module):
    """Bidirectional LSTM with attention for sequence-level ECG classification.

    The model treats a sliding window of consecutive beats as a temporal
    sequence and classifies the *centre* beat using contextual information
    from surrounding beats.

    Architecture
    ------------
    * Input projection: each beat (129 samples) → linear embedding
    * Bidirectional LSTM: 2 layers, hidden_size=128  → 256 per step
    * Additive attention over LSTM outputs
    * Dropout(0.5)
    * FC(256→128) → ReLU → FC(128→num_classes)

    Parameters
    ----------
    num_classes:
        Number of output classes (default 5).
    beat_len:
        Number of samples per beat (default 129).
    seq_len:
        Number of consecutive beats per sequence (default 10).
    hidden_size:
        LSTM hidden state size per direction (default 128).
    num_layers:
        Number of stacked LSTM layers (default 2).
    dropout:
        Dropout probability (default 0.5).
    """

    def __init__(
        self,
        num_classes: int = 5,
        beat_len: int = 129,
        seq_len: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.beat_len = beat_len
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Project each beat to a lower-dimensional embedding before LSTM
        self.input_proj = nn.Sequential(
            nn.Linear(beat_len, 64),
            nn.ReLU(),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * 2  # bidirectional → 256

        self.attention = _BeatAttention(lstm_out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(lstm_out_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, beat_len)``.
        return_attention : bool
            If ``True``, return ``(logits, attention_weights)`` instead of
            just logits.

        Returns
        -------
        logits : torch.Tensor, shape ``(batch, num_classes)``
        attention_weights : torch.Tensor, shape ``(batch, seq_len)``
            Only returned when *return_attention* is ``True``.
        """
        batch, seq, feat = x.shape  # (B, T, 129)

        # Project each beat independently
        x_proj = self.input_proj(x.view(batch * seq, feat))      # (B*T, 64)
        x_proj = x_proj.view(batch, seq, -1)                      # (B, T, 64)

        # LSTM
        lstm_out, _ = self.lstm(x_proj)                           # (B, T, 256)

        # Attention
        context, attn_weights = self.attention(lstm_out)           # (B, 256), (B, T)

        # Classification head
        out = self.dropout(context)
        out = F.relu(self.fc1(out))
        logits = self.fc2(out)

        if return_attention:
            return logits, attn_weights
        return logits

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention weights for a batch of sequences.

        Parameters
        ----------
        x : torch.Tensor, shape ``(batch, seq_len, beat_len)``

        Returns
        -------
        torch.Tensor, shape ``(batch, seq_len)``
        """
        _, weights = self.forward(x, return_attention=True)
        return weights
