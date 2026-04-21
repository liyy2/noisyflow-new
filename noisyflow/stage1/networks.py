from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn

from noisyflow.nn import MLP


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B, 1) or (B,)
        returns: (B, dim)
        """
        if t.dim() == 1:
            t = t[:, None]
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0, math.log(10000.0), half, device=t.device)
        )  # (half,)
        angles = t * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb


class VelocityField(nn.Module):
    """
    f_psi(z, t, label, cond) -> velocity in R^d.
    """

    def __init__(
        self,
        d: int,
        num_classes: int,
        hidden: List[int] = [256, 256, 256],
        time_emb_dim: int = 64,
        label_emb_dim: int = 64,
        cond_dim: int = 0,
        cond_emb_dim: int = 0,
        act: str = "silu",
        mlp_norm: str = "none",
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.num_classes = num_classes
        self.cond_dim = int(cond_dim)
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.label_emb = nn.Embedding(num_classes, label_emb_dim)
        cond_emb_dim = int(cond_emb_dim)
        if self.cond_dim < 0:
            raise ValueError("cond_dim must be >= 0")
        if cond_emb_dim < 0:
            raise ValueError("cond_emb_dim must be >= 0")
        self.cond_proj = None
        cond_out = 0
        if self.cond_dim > 0:
            cond_out = cond_emb_dim if cond_emb_dim > 0 else self.cond_dim
            self.cond_proj = nn.Linear(self.cond_dim, cond_out)
        in_dim = d + time_emb_dim + label_emb_dim + cond_out
        self.mlp = MLP(in_dim, d, hidden=hidden, act=act, norm=mlp_norm, dropout=mlp_dropout)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        z: (B, d)
        t: (B, 1) or (B,)
        y: (B,) int64 labels
        cond: optional (B, cond_dim) continuous covariates (e.g., timepoint, batch embedding, spatial coords)
        """
        te = self.time_emb(t)  # (B, time_emb_dim)
        le = self.label_emb(y)  # (B, label_emb_dim)
        blocks = [z, te, le]
        if self.cond_proj is not None:
            if cond is None:
                cond = torch.zeros((z.shape[0], self.cond_dim), device=z.device, dtype=z.dtype)
            else:
                if cond.dim() == 1:
                    cond = cond[:, None]
                cond = cond.to(device=z.device, dtype=z.dtype)
                if int(cond.shape[0]) != int(z.shape[0]):
                    raise ValueError(f"cond must have batch size {z.shape[0]}, got {cond.shape[0]}")
                if int(cond.shape[1]) != int(self.cond_dim):
                    raise ValueError(f"cond must have shape (B,{self.cond_dim}), got {tuple(cond.shape)}")
            blocks.append(self.cond_proj(cond))
        h = torch.cat(blocks, dim=-1)
        return self.mlp(h)


class ConditionalVAE(nn.Module):
    """
    Class-conditional VAE for Stage I synthetic sample generation.
    """

    def __init__(
        self,
        d: int,
        num_classes: int,
        hidden: List[int] = [256, 256, 256],
        latent_dim: int = 32,
        label_emb_dim: int = 64,
        cond_dim: int = 0,
        cond_emb_dim: int = 0,
        act: str = "silu",
        mlp_norm: str = "none",
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.d = int(d)
        self.num_classes = int(num_classes)
        self.latent_dim = int(latent_dim)
        self.cond_dim = int(cond_dim)
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if self.cond_dim < 0:
            raise ValueError("cond_dim must be >= 0")
        cond_emb_dim = int(cond_emb_dim)
        if cond_emb_dim < 0:
            raise ValueError("cond_emb_dim must be >= 0")

        self.label_emb = nn.Embedding(self.num_classes, int(label_emb_dim))
        self.cond_proj = None
        cond_out = 0
        if self.cond_dim > 0:
            cond_out = cond_emb_dim if cond_emb_dim > 0 else self.cond_dim
            self.cond_proj = nn.Linear(self.cond_dim, cond_out)

        encoder_in_dim = self.d + int(label_emb_dim) + cond_out
        decoder_in_dim = self.latent_dim + int(label_emb_dim) + cond_out
        self.encoder = MLP(
            encoder_in_dim,
            2 * self.latent_dim,
            hidden=hidden,
            act=act,
            norm=mlp_norm,
            dropout=mlp_dropout,
        )
        self.decoder = MLP(
            decoder_in_dim,
            self.d,
            hidden=hidden,
            act=act,
            norm=mlp_norm,
            dropout=mlp_dropout,
        )

    def _condition_blocks(
        self,
        y: torch.Tensor,
        cond: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        blocks = [self.label_emb(y)]
        if self.cond_proj is not None:
            if cond is None:
                cond = torch.zeros((batch_size, self.cond_dim), device=device, dtype=dtype)
            else:
                if cond.dim() == 1:
                    cond = cond[:, None]
                cond = cond.to(device=device, dtype=dtype)
                if int(cond.shape[0]) != int(batch_size):
                    raise ValueError(f"cond must have batch size {batch_size}, got {cond.shape[0]}")
                if int(cond.shape[1]) != int(self.cond_dim):
                    raise ValueError(f"cond must have shape (B,{self.cond_dim}), got {tuple(cond.shape)}")
            blocks.append(self.cond_proj(cond))
        return blocks

    def encode(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = [x]
        blocks.extend(
            self._condition_blocks(y, cond, batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        )
        h = torch.cat(blocks, dim=-1)
        params = self.encoder(h)
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        blocks = [z]
        blocks.extend(
            self._condition_blocks(y, cond, batch_size=z.shape[0], device=z.device, dtype=z.dtype)
        )
        h = torch.cat(blocks, dim=-1)
        return self.decoder(h)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y, cond=cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y, cond=cond)
        return recon, mu, logvar
