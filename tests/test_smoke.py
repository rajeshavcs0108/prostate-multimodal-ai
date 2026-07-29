"""Smoke tests: verify the model builds and runs a forward pass, and that seeding is
deterministic. Run with: pytest tests/"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.framework import InterpretableMultimodalFramework, composite_loss  # noqa: E402
from seed import set_seed  # noqa: E402


def _default_model_cfg():
    return {
        "latent_dim": 32,
        "encoders": {
            "imaging": {"hidden_dims": [16], "activation": "gelu", "dropout": 0.0},
            "clinical": {"hidden_dims": [16], "activation": "gelu", "dropout": 0.0},
            "molecular": {"hidden_dims": [16], "activation": "gelu", "dropout": 0.0},
        },
        "attention_fusion": {"n_heads": 2, "attn_dropout": 0.0},
        "prediction_head": {"type": "linear_sigmoid"},
    }


def test_forward_pass_shapes():
    set_seed(42)
    in_dims = {"imaging": 7, "clinical": 10, "molecular": 50}
    model = InterpretableMultimodalFramework(in_dims, _default_model_cfg())

    b = 4
    imaging = torch.randn(b, in_dims["imaging"])
    clinical = torch.randn(b, in_dims["clinical"])
    molecular = torch.randn(b, in_dims["molecular"])

    out = model(imaging, clinical, molecular)
    assert out["y_hat"].shape == (b,)
    assert torch.all((out["y_hat"] >= 0) & (out["y_hat"] <= 1))
    assert out["attn_weights"].shape == (b, 3, 3)


def test_composite_loss_is_finite():
    set_seed(42)
    in_dims = {"imaging": 7, "clinical": 10, "molecular": 50}
    model = InterpretableMultimodalFramework(in_dims, _default_model_cfg())

    b = 4
    imaging = torch.randn(b, in_dims["imaging"])
    clinical = torch.randn(b, in_dims["clinical"])
    molecular = torch.randn(b, in_dims["molecular"])
    labels = torch.randint(0, 2, (b,)).float()

    out = model(imaging, clinical, molecular)
    loss_cfg = {"bce_weight": 1.0, "sparsity_weight": 0.01, "attention_reg_weight": 0.005}
    losses = composite_loss(out["y_hat"], labels, model, out["attn_weights"], loss_cfg)
    assert torch.isfinite(losses["total"])


def test_seed_determinism():
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.allclose(a, b)
