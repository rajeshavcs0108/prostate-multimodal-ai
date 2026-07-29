"""Generates synthetic, clearly-labeled demo data so reviewers can run the full pipeline
(train -> evaluate -> infer -> explain) without needing access to the real, ethics-approved
patient cohort. This data is NOT real and must never be cited as representing the study's
actual results.

Usage:
    python scripts/make_synthetic_demo_data.py --n_patients 400 --out_dir data/synthetic
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_patients", type=int, default=400)
    parser.add_argument("--imaging_dim", type=int, default=7)   # matches extract_radiomics() feature count
    parser.add_argument("--clinical_dim", type=int, default=10)
    parser.add_argument("--molecular_dim", type=int, default=50)
    parser.add_argument("--out_dir", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    n = args.n_patients
    labels = rng.integers(0, 2, size=n).astype(np.float32)

    # Give each modality a mild, label-correlated signal plus noise, so the pipeline has
    # something learnable end-to-end (purely for smoke-testing, not scientific validity).
    def synth_block(dim, label_weight):
        base = rng.normal(0, 1, size=(n, dim)).astype(np.float32)
        signal = np.outer(labels, rng.normal(0, label_weight, size=dim)).astype(np.float32)
        return base + signal

    imaging = synth_block(args.imaging_dim, 0.6)
    clinical = synth_block(args.clinical_dim, 0.6)
    molecular = synth_block(args.molecular_dim, 0.4)

    np.save(os.path.join(args.out_dir, "imaging.npy"), imaging)
    np.save(os.path.join(args.out_dir, "clinical.npy"), clinical)
    np.save(os.path.join(args.out_dir, "molecular.npy"), molecular)
    np.save(os.path.join(args.out_dir, "labels.npy"), labels)

    print(f"Synthetic demo data written to {args.out_dir} ({n} patients). NOT real patient data.")


if __name__ == "__main__":
    main()
