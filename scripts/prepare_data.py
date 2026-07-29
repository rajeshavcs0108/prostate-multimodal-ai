"""Dataset preparation pipeline: raw cohort -> processed model-ready tensors.

Defines the exact expected schema for the real patient cohort described in the manuscript
(Section IV, Experimental Setup), so this script can be pointed directly at IRB-approved
data once available. It is not run automatically since no raw data ships with this repo.

Expected raw_dir layout:

    raw_dir/
      imaging/
        <patient_id>_mri.npy        # multiparametric MRI volume
        <patient_id>_mask.npy       # tumor segmentation mask (same shape as MRI)
      clinical.csv                  # columns: patient_id, age, cancer_stage, gleason_score, psa_level, ...
      molecular.csv                 # columns: patient_id, gene_marker_1, gene_marker_2, ...
      labels.csv                    # columns: patient_id, aggressiveness_label (0/1)

Usage:
    python scripts/prepare_data.py --raw_dir data/raw --out_dir data/processed
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data.preprocessing import (  # noqa: E402
    ClinicalPreprocessor,
    apply_tumor_mask,
    extract_radiomics,
    normalize_intensity,
    select_molecular_features,
)


def build_imaging_features(raw_dir: str, patient_ids: list[str]) -> np.ndarray:
    rows = []
    for pid in patient_ids:
        mri = np.load(os.path.join(raw_dir, "imaging", f"{pid}_mri.npy"))
        mask = np.load(os.path.join(raw_dir, "imaging", f"{pid}_mask.npy"))
        mri = normalize_intensity(mri)
        masked = apply_tumor_mask(mri, mask)
        feats = extract_radiomics(masked)
        rows.append(list(feats.values()))
    return np.array(rows, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_k_molecular", type=int, default=50)
    parser.add_argument("--variance_threshold", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    labels_df = pd.read_csv(os.path.join(args.raw_dir, "labels.csv"))
    clinical_df = pd.read_csv(os.path.join(args.raw_dir, "clinical.csv")).merge(labels_df, on="patient_id")
    molecular_df = pd.read_csv(os.path.join(args.raw_dir, "molecular.csv")).merge(labels_df, on="patient_id")

    patient_ids = clinical_df["patient_id"].tolist()
    labels = clinical_df["aggressiveness_label"].to_numpy(dtype=np.float32)

    # Imaging
    imaging_feats = build_imaging_features(args.raw_dir, patient_ids)

    # Clinical
    continuous_fields = ["age", "gleason_score", "psa_level"]
    categorical_fields = ["cancer_stage"]
    clin_prep = ClinicalPreprocessor(continuous_fields, categorical_fields)
    clinical_feats = clin_prep.fit_transform(clinical_df)

    # Molecular
    molecular_cols = [c for c in molecular_df.columns if c not in ("patient_id", "aggressiveness_label")]
    molecular_raw = molecular_df[molecular_cols].to_numpy(dtype=np.float32)
    molecular_feats, selected_idx = select_molecular_features(
        molecular_raw, labels, variance_threshold=args.variance_threshold, top_k=args.top_k_molecular
    )

    np.save(os.path.join(args.out_dir, "imaging.npy"), imaging_feats)
    np.save(os.path.join(args.out_dir, "clinical.npy"), clinical_feats)
    np.save(os.path.join(args.out_dir, "molecular.npy"), molecular_feats)
    np.save(os.path.join(args.out_dir, "labels.npy"), labels)
    np.save(os.path.join(args.out_dir, "molecular_selected_idx.npy"), selected_idx)

    print(f"Processed {len(patient_ids)} patients -> {args.out_dir}")
    print(f"  imaging:   {imaging_feats.shape}")
    print(f"  clinical:  {clinical_feats.shape}")
    print(f"  molecular: {molecular_feats.shape}")


if __name__ == "__main__":
    main()
