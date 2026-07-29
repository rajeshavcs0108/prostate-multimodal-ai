"""Single-patient / batch inference with intrinsic + attention explanations.

Usage:
    python src/infer.py --config configs/config.yaml --patient_id 0001
    python src/infer.py --config configs/config.yaml --batch  # runs on the full processed set
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import MultimodalProstateDataset
from explainability.attention_maps import modality_importance_from_attention
from models.framework import InterpretableMultimodalFramework
from train import get_device


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_best_model(cfg: dict, device: str, seed: int | None = None) -> InterpretableMultimodalFramework:
    seed = seed or cfg["seed"]["run_seeds"][0]
    ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"run_{seed}", "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Train first with `python src/train.py --config {cfg}`."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    model = InterpretableMultimodalFramework(ckpt["in_dims"], ckpt["cfg"]["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, imaging: torch.Tensor, clinical: torch.Tensor, molecular: torch.Tensor, device: str) -> dict:
    out = model(imaging.unsqueeze(0).to(device), clinical.unsqueeze(0).to(device), molecular.unsqueeze(0).to(device))
    prob = float(out["y_hat"].item())
    modality_importance = modality_importance_from_attention(out["attn_weights"])[0]
    return {
        "aggressiveness_probability": prob,
        "predicted_label": "aggressive" if prob >= 0.5 else "non-aggressive",
        "modality_importance": {
            "imaging": float(modality_importance[0]),
            "clinical": float(modality_importance[1]),
            "molecular": float(modality_importance[2]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--patient_id", default=None, help="Row index into the processed dataset")
    parser.add_argument("--batch", action="store_true", help="Run inference on the entire dataset")
    parser.add_argument("--seed", type=int, default=None, help="Which of the 5 trained runs' checkpoint to use")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg["data"].get("synthetic_dir", cfg["data"]["processed_dir"])
    device = get_device(cfg["training"]["device"])

    ds = MultimodalProstateDataset.from_dir(data_dir)
    model = load_best_model(cfg, device, args.seed)

    if args.batch:
        results = []
        for i in range(len(ds)):
            item = ds[i]
            results.append(predict(model, item["imaging"], item["clinical"], item["molecular"], device))
        os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
        import json
        out_path = os.path.join(cfg["paths"]["results_dir"], "batch_predictions.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {len(results)} predictions to {out_path}")
    else:
        idx = int(args.patient_id) if args.patient_id is not None else 0
        item = ds[idx]
        result = predict(model, item["imaging"], item["clinical"], item["molecular"], device)
        print(result)


if __name__ == "__main__":
    main()
