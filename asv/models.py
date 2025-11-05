"""Neural network models used by the agent."""
from __future__ import annotations

import torch
import torch.nn as nn


class QNet(nn.Module):
    """Feed-forward network used for action-value estimation."""

    def __init__(self, in_dim: int = 100, n_actions: int = 9) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_actions),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.net(x)
