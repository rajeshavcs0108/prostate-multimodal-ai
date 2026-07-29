"""Attention-map extraction and plotting for modality-level interpretability.

Consumes the `attn_weights` tensor returned by `CrossModalAttentionFusion.forward`
(shape B x 3 x 3, tokens ordered [imaging, clinical, molecular]) to produce the
per-patient and population-level modality importance visualizations described
alongside Figure 9 (SHAP) as complementary intrinsic explanations.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

MODALITY_NAMES = ["Imaging", "Clinical", "Molecular"]


def modality_importance_from_attention(attn_weights: torch.Tensor) -> np.ndarray:
    """attn_weights: (B, 3, 3) -> (B, 3) per-modality importance (row-averaged attention
    received by each modality token)."""
    importance = attn_weights.mean(dim=1)  # average over query positions
    return importance.detach().cpu().numpy()


def plot_population_attention(attn_weights: torch.Tensor, out_path: str) -> None:
    importance = modality_importance_from_attention(attn_weights)
    means = importance.mean(axis=0)
    stds = importance.std(axis=0)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(MODALITY_NAMES, means, yerr=stds, capsize=4)
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Population-level modality importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_patient_attention(attn_matrix: np.ndarray, patient_id: str, out_path: str) -> None:
    """attn_matrix: (3, 3) single-patient attention map."""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(attn_matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(MODALITY_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(MODALITY_NAMES)
    ax.set_title(f"Attention map — patient {patient_id}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
