"""Post-hoc SHAP-based feature attribution (Section III, Figure 9).

Wraps the trained multimodal model as a single flattened-input function so `shap.KernelExplainer`
(model-agnostic, works for any architecture) can attribute predictions to individual imaging,
clinical, and molecular features.
"""
from __future__ import annotations

import numpy as np
import shap
import torch


def _make_predict_fn(model, dims: dict[str, int], device: str):
    i_end = dims["imaging"]
    c_end = i_end + dims["clinical"]

    def predict_fn(x: np.ndarray) -> np.ndarray:
        model.eval()
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        imaging, clinical, molecular = x_t[:, :i_end], x_t[:, i_end:c_end], x_t[:, c_end:]
        with torch.no_grad():
            out = model(imaging, clinical, molecular)
        return out["y_hat"].cpu().numpy()

    return predict_fn


def compute_shap_values(model, background: np.ndarray, samples: np.ndarray, dims: dict[str, int],
                         device: str = "cpu", n_background: int = 100, n_explain: int = 200):
    """Returns SHAP values of shape (n_explain, n_features) for `samples`, using `background`
    (typically the training set) to estimate the baseline expectation E[f(x)]."""
    rng = np.random.default_rng(42)

    bg_idx = rng.choice(len(background), size=min(n_background, len(background)), replace=False)
    background_sample = background[bg_idx]

    explain_idx = rng.choice(len(samples), size=min(n_explain, len(samples)), replace=False)
    explain_sample = samples[explain_idx]

    predict_fn = _make_predict_fn(model, dims, device)
    explainer = shap.KernelExplainer(predict_fn, background_sample)
    shap_values = explainer.shap_values(explain_sample, nsamples="auto")
    return shap_values, explain_idx


def feature_importance_ranking(shap_values: np.ndarray, feature_names: list[str]) -> list[tuple[str, float]]:
    """Mean |SHAP value| per feature, ranked descending — feeds Figure 9's bar plot."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(feature_names, mean_abs), key=lambda t: t[1], reverse=True)
    return ranking
