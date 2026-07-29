"""Full interpretable multimodal AI framework (Algorithm 1 / Section III).

Model:  X_img, X_clin, X_mol -> encoders -> attention fusion -> linear+sigmoid head -> y_hat

Loss:   L = L_bce(y, y_hat) + lambda_sparse * ||W_p||_1 + lambda_attn * H_reg(attention)

The sparsity term encourages compact, more interpretable prediction weights; the
attention-regularization term discourages overly diffuse (uniform) attention, so the
learned modality weighting stays informative for the intrinsic-interpretability analysis.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_fusion import CrossModalAttentionFusion
from .encoders import build_encoder


class InterpretableMultimodalFramework(nn.Module):
    def __init__(self, in_dims: dict[str, int], model_cfg: dict):
        super().__init__()
        latent_dim = model_cfg["latent_dim"]

        self.encoder_img = build_encoder(in_dims["imaging"], latent_dim, model_cfg["encoders"]["imaging"])
        self.encoder_clin = build_encoder(in_dims["clinical"], latent_dim, model_cfg["encoders"]["clinical"])
        self.encoder_mol = build_encoder(in_dims["molecular"], latent_dim, model_cfg["encoders"]["molecular"])

        self.fusion = CrossModalAttentionFusion(
            latent_dim=latent_dim,
            n_heads=model_cfg["attention_fusion"]["n_heads"],
            attn_dropout=model_cfg["attention_fusion"]["attn_dropout"],
        )

        self.prediction_head = nn.Linear(latent_dim, 1)

    def forward(self, imaging: torch.Tensor, clinical: torch.Tensor, molecular: torch.Tensor):
        z_img = self.encoder_img(imaging)
        z_clin = self.encoder_clin(clinical)
        z_mol = self.encoder_mol(molecular)

        z_fused, attn_weights = self.fusion(z_img, z_clin, z_mol)

        logits = self.prediction_head(z_fused).squeeze(-1)
        y_hat = torch.sigmoid(logits)

        return {
            "y_hat": y_hat,
            "logits": logits,
            "attn_weights": attn_weights,
            "z_fused": z_fused,
            "modality_embeddings": {"imaging": z_img, "clinical": z_clin, "molecular": z_mol},
        }

    def modality_weight_importance(self) -> torch.Tensor:
        """Intrinsic interpretability: |W_p| magnitude per latent dimension, used together
        with attention weights to estimate modality-level importance (Section III)."""
        return self.prediction_head.weight.detach().abs().squeeze(0)


def composite_loss(y_hat: torch.Tensor, y_true: torch.Tensor, model: InterpretableMultimodalFramework,
                    attn_weights: torch.Tensor, loss_cfg: dict) -> dict[str, torch.Tensor]:
    bce = F.binary_cross_entropy(y_hat, y_true)

    sparsity = model.prediction_head.weight.abs().sum()

    # Attention regularization: penalize low-entropy... wait, we want to *discourage* overly
    # diffuse (high-entropy/uniform) attention, so we penalize entropy directly.
    eps = 1e-8
    attn_entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1).mean()

    total = (
        loss_cfg["bce_weight"] * bce
        + loss_cfg["sparsity_weight"] * sparsity
        + loss_cfg["attention_reg_weight"] * attn_entropy
    )
    return {"total": total, "bce": bce, "sparsity": sparsity, "attn_entropy": attn_entropy}
