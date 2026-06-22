from __future__ import annotations

import torch
from torch import nn


class FallBiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 132,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.feature_proj = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
        )

        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Nhan x shape (batch, 75, 132), tra ve logits shape (batch, 2)."""
        x = self.feature_proj(x)
        lstm_out, _ = self.bilstm(x)
        last_frame = lstm_out[:, -1, :]
        logits = self.classifier(last_frame)
        return logits

    def count_parameters(self) -> int:
        """Tra ve so tham so co the train."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
