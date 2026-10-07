from __future__ import annotations

import torch
from torch import nn

from config import THICKNESS_MIN_NM, THICKNESS_MAX_NM


class ThinFilmMLP(nn.Module):
    """4 -> 128 -> 128 -> 64 -> 41 MLP surrogate model."""

    def __init__(self) -> None:
        super().__init__()
        center = (THICKNESS_MIN_NM + THICKNESS_MAX_NM) / 2.0
        scale = (THICKNESS_MAX_NM - THICKNESS_MIN_NM) / 2.0
        self.register_buffer("input_center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("input_scale", torch.tensor(scale, dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 41),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_center) / self.input_scale
        return self.net(x)
