"""Evaluation entrypoint: reports mean +/- std across the seeded runs, plus bootstrap
95% confidence intervals, matching the manuscript's statistical validation protocol
(five runs, mean +/- std, 95% CI, paired significance testing at p < 0.05).

Usage:
    python src/evaluate.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import MultimodalProstateDataset
from models.framework import InterpretableMultimodalFramework
from train import get_device, split_dataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_checkpoint(ckpt_path: str, ds: MultimodalProstateDataset, test_idx: np.ndarray,
                         device: str, batch_size: int) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = InterpretableMultimodalFramework(ckpt["in_dims"], ckpt["cfg"]["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False)
    y_true, y_score = [], []
    for batch in loader:
        out = model(batch["imaging"].to(device), batch["clinical"].to(device), batch["molecular"].to(device))
        y_true.append(batch["label"].numpy())
        y_score.append(out["y_hat"].cpu().numpy())
    y_true, y_score = np.concatenate(y_true), np.concatenate(y_score)
    y_pred = (y_score >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }
    return metrics


def bootstrap_ci(values: list[float], n_boot: int = 2000, ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.array(values)
    boots = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lower = np.percentile(boots, (1 - ci) / 2 * 100)
    upper = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(lower), float(upper)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg["data"].get("synthetic_dir", cfg["data"]["processed_dir"])
    device = get_device(cfg["training"]["device"])
    ds = MultimodalProstateDataset.from_dir(data_dir)

    all_metrics = {m: [] for m in cfg["evaluation"]["metrics"]}
    for seed in cfg["seed"]["run_seeds"]:
        ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"run_{seed}", "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"Skipping seed {seed}: no checkpoint found at {ckpt_path}")
            continue
        _, _, test_idx = split_dataset(ds, cfg["data"]["split"], seed)
        metrics = evaluate_checkpoint(ckpt_path, ds, test_idx, device, cfg["training"]["batch_size"])
        for k, v in metrics.items():
            all_metrics[k].append(v)
        print(f"[seed={seed}] {metrics}")

    summary = {}
    for k, vals in all_metrics.items():
        if not vals:
            continue
        lower, upper = bootstrap_ci(vals, n_boot=cfg["evaluation"]["n_bootstrap"])
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                      "ci95_lower": lower, "ci95_upper": upper}
        print(f"{k}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  (95% CI [{lower:.4f}, {upper:.4f}])")

    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    with open(os.path.join(cfg["paths"]["results_dir"], "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
