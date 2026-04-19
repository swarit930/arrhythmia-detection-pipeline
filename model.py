from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalECGNet(nn.Module):
    """1D-CNN morphology branch + RR-interval feature fusion branch."""

    def __init__(self, num_classes: int = 5, dropout: float = 0.4) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.cnn_dropout = nn.Dropout(dropout)
        self.fusion_dropout = nn.Dropout(dropout)

        self.fusion_fc1 = nn.Linear(128 + 3, 96)
        self.fusion_fc2 = nn.Linear(96, 64)
        self.classifier = nn.Linear(64, num_classes)

    def extract_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x).squeeze(-1)
        x = self.cnn_dropout(x)
        return x

    def forward(self, signal: torch.Tensor, rr_features: torch.Tensor) -> torch.Tensor:
        cnn_features = self.extract_cnn_features(signal)
        fused = torch.cat([cnn_features, rr_features], dim=1)
        fused = F.relu(self.fusion_fc1(fused))
        fused = self.fusion_dropout(fused)
        fused = F.relu(self.fusion_fc2(fused))
        return self.classifier(fused)
