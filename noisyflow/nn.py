from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: List[int],
        act: str = "silu",
        norm: str = "none",
        dropout: float = 0.0,
    ):
        super().__init__()
        acts = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "softplus": nn.Softplus(),
        }
        if act not in acts:
            raise ValueError(f"Unknown activation {act}")
        norm = str(norm).lower()
        if norm not in {"none", "layer"}:
            raise ValueError("norm must be 'none' or 'layer'")
        dropout = float(dropout)
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if norm == "layer":
                layers.append(nn.LayerNorm(h))
            layers.append(acts[act])
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
