from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """MLP value head for RL: maps hidden states to scalar values."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map hidden states to scalar values.

        Args:
            hidden: (batch, seq_len, d_model)

        Returns:
            values: (batch, seq_len)
        """
        return self.net(hidden).squeeze(-1)
