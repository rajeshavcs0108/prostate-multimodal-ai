#!/usr/bin/env bash
# One-command demo pipeline: synthetic data -> train -> evaluate -> infer.
# For the real cohort, replace step 1 with scripts/prepare_data.py and point
# configs/config.yaml at data/processed instead of data/synthetic.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/4] Generating synthetic demo data..."
python scripts/make_synthetic_demo_data.py

echo "[2/4] Training (5 seeded runs)..."
python src/train.py --config configs/config.yaml

echo "[3/4] Evaluating..."
python src/evaluate.py --config configs/config.yaml

echo "[4/4] Running batch inference..."
python src/infer.py --config configs/config.yaml --batch

echo "Done. See results/ for metrics and predictions."
