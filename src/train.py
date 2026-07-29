"""Training entrypoint.

Usage:
    python src/train.py --config configs/config.yaml

Runs `n_runs` independent, individually-seeded training runs (paper: "five independent
runs of the experiment"), each with early stopping on validation AUC, and saves the
best checkpoint per run under `checkpoints/run_<seed>/best_model.pt`.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import MultimodalProstateDataset
from models.framework import InterpretableMultimodalFramework, composite_loss
from seed import set_seed


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(cfg_device: str) -> str:
    if cfg_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def split_dataset(ds: MultimodalProstateDataset, split_cfg: dict, seed: int):
    n = len(ds)
    idx = np.arange(n)
    labels = ds.labels.numpy()

    train_idx, temp_idx = train_test_split(
        idx, test_size=(1 - split_cfg["train"]), random_state=seed,
        stratify=labels if split_cfg.get("stratify", True) else None,
    )
    rel_val = split_cfg["val"] / (split_cfg["val"] + split_cfg["test"])
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - rel_val), random_state=seed,
        stratify=labels[temp_idx] if split_cfg.get("stratify", True) else None,
    )
    return train_idx, val_idx, test_idx


def run_one_seed(cfg: dict, seed: int, ds: MultimodalProstateDataset, device: str) -> dict:
    set_seed(seed)

    train_idx, val_idx, test_idx = split_dataset(ds, cfg["data"]["split"], seed)
    train_loader = DataLoader(torch.utils.data.Subset(ds, train_idx),
                               batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(ds, val_idx),
                             batch_size=cfg["training"]["batch_size"], shuffle=False)

    in_dims = {
        "imaging": ds.imaging.shape[1],
        "clinical": ds.clinical.shape[1],
        "molecular": ds.molecular.shape[1],
    }
    model = InterpretableMultimodalFramework(in_dims, cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                   weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["max_epochs"])

    best_val_auc = -1.0
    epochs_no_improve = 0
    ckpt_dir = os.path.join(cfg["paths"]["checkpoint_dir"], f"run_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(cfg["training"]["max_epochs"]):
        model.train()
        for batch in train_loader:
            imaging = batch["imaging"].to(device)
            clinical = batch["clinical"].to(device)
            molecular = batch["molecular"].to(device)
            label = batch["label"].to(device)

            out = model(imaging, clinical, molecular)
            losses = composite_loss(out["y_hat"], label, model, out["attn_weights"], cfg["loss"])

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            optimizer.step()
        scheduler.step()

        val_auc = _evaluate_auc(model, val_loader, device)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "in_dims": in_dims, "cfg": cfg},
                       os.path.join(ckpt_dir, "best_model.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["training"]["early_stopping_patience"]:
                break

    return {"seed": seed, "best_val_auc": best_val_auc, "test_idx": test_idx, "checkpoint": ckpt_dir}


@torch.no_grad()
def _evaluate_auc(model, loader, device) -> float:
    from sklearn.metrics import roc_auc_score
    model.eval()
    y_true, y_score = [], []
    for batch in loader:
        out = model(batch["imaging"].to(device), batch["clinical"].to(device), batch["molecular"].to(device))
        y_true.append(batch["label"].numpy())
        y_score.append(out["y_hat"].cpu().numpy())
    y_true, y_score = np.concatenate(y_true), np.concatenate(y_score)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data_dir", default=None, help="Override processed data directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg["data"].get("synthetic_dir", cfg["data"]["processed_dir"])
    device = get_device(cfg["training"]["device"])
    print(f"Using device: {device}")

    ds = MultimodalProstateDataset.from_dir(data_dir)

    results = []
    for seed in tqdm(cfg["seed"]["run_seeds"], desc="Independent runs"):
        res = run_one_seed(cfg, seed, ds, device)
        print(f"[seed={seed}] best val AUC = {res['best_val_auc']:.4f}")
        results.append(res)

    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    import json
    with open(os.path.join(cfg["paths"]["results_dir"], "train_summary.json"), "w") as f:
        json.dump([{"seed": r["seed"], "best_val_auc": r["best_val_auc"]} for r in results], f, indent=2)


if __name__ == "__main__":
    main()
