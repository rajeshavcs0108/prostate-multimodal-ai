"""Modality-specific encoders (f_img, f_clin, f_mol) projecting each modality into the
shared latent embedding space of dimension `latent_dim`, per Section III."""
from __future__ import annotations

import torch
import torch.nn as nn

_ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], latent_dim: int, activation: str = "gelu", dropout: float = 0.2):
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation, nn.GELU)
        dims = [in_dim, *hidden_dims, latent_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last:
                layers.append(act_cls())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_encoder(in_dim: int, latent_dim: int, cfg: dict) -> ModalityEncoder:
    return ModalityEncoder(
        in_dim=in_dim,
        hidden_dims=cfg["hidden_dims"],
        latent_dim=latent_dim,
        activation=cfg.get("activation", "gelu"),
        dropout=cfg.get("dropout", 0.2),
    )
