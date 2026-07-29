"""Cross-modal self-attention fusion, per Section III:

    Z = [Z_img; Z_clin; Z_mol]                 (concatenated modality embeddings)
    Q, K, V = Z W_Q, Z W_K, Z W_V
    A = softmax(Q K^T / sqrt(d))
    Z_f = A V

Implemented as standard multi-head self-attention over the 3 modality tokens, which
recovers the manuscript's single-head formulation when `n_heads=1` and additionally
supports multi-head attention for richer cross-modal interaction modeling.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, latent_dim: int, n_heads: int = 4, attn_dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=n_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z_img: torch.Tensor, z_clin: torch.Tensor, z_mol: torch.Tensor):
        """
        Args:
            z_img, z_clin, z_mol: (B, latent_dim) modality embeddings.
        Returns:
            z_fused: (B, latent_dim) attention-fused representation (mean-pooled over modality tokens).
            attn_weights: (B, n_modalities, n_modalities) attention map for interpretability
                          (Figure-9-style modality/attention visualization).
        """
        z = torch.stack([z_img, z_clin, z_mol], dim=1)  # (B, 3, latent_dim)
        fused_tokens, attn_weights = self.mha(z, z, z, need_weights=True, average_attn_weights=True)
        fused_tokens = self.norm(fused_tokens + z)  # residual connection
        z_fused = fused_tokens.mean(dim=1)  # pool the 3 modality tokens -> (B, latent_dim)
        return z_fused, attn_weights

    @staticmethod
    def modality_importance(attn_weights: torch.Tensor) -> torch.Tensor:
        """Aggregate attention weights into a per-modality importance score (B, 3):
        imaging, clinical, molecular — used for the attention-map visualizations."""
        return attn_weights.mean(dim=1)  # average over query tokens
