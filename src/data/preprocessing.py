"""Preprocessing pipeline for the three modalities, implementing Section III of the manuscript.

- Imaging: intensity normalization, tumor segmentation mask application, radiomic
  feature extraction (R = phi(X_img ⊙ M)).
- Clinical: z-score normalization of continuous variables, one-hot encoding of
  categoricals.
- Molecular: variance filtering followed by mutual-information ranking, keeping the
  top-K features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------------------------------------------
# Imaging
# --------------------------------------------------------------------------------------
def normalize_intensity(volume: np.ndarray) -> np.ndarray:
    """Z-score normalize an imaging volume to reduce scanner-related variability."""
    mu, sigma = volume.mean(), volume.std() + 1e-8
    return (volume - mu) / sigma


def apply_tumor_mask(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Retain only voxels within the tumor mask: X_img ⊙ M."""
    if volume.shape != mask.shape:
        raise ValueError(f"Shape mismatch: volume {volume.shape} vs mask {mask.shape}")
    return volume * mask


def extract_radiomics(masked_volume: np.ndarray) -> dict[str, float]:
    """Extract first-order, shape, and texture (GLCM/GLRLM-style) descriptors.

    This is a lightweight, dependency-free stand-in for a full radiomics toolkit
    (e.g. pyradiomics). Swap this function out for `pyradiomics.featureextractor`
    when running on the real cohort for publication-grade radiomic features.
    """
    voxels = masked_volume[masked_volume != 0]
    if voxels.size == 0:
        voxels = np.zeros(1)

    features = {
        "first_order_mean": float(voxels.mean()),
        "first_order_std": float(voxels.std()),
        "first_order_skewness": float(_skewness(voxels)),
        "first_order_kurtosis": float(_kurtosis(voxels)),
        "shape_volume_voxels": float((masked_volume != 0).sum()),
        "texture_energy": float(np.mean(voxels ** 2)),
        "texture_entropy": float(_entropy(voxels)),
    }
    return features


def _skewness(x: np.ndarray) -> float:
    mu, sigma = x.mean(), x.std() + 1e-8
    return float(np.mean(((x - mu) / sigma) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    mu, sigma = x.mean(), x.std() + 1e-8
    return float(np.mean(((x - mu) / sigma) ** 4) - 3.0)


def _entropy(x: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log(hist)))


# --------------------------------------------------------------------------------------
# Clinical
# --------------------------------------------------------------------------------------
@dataclass
class ClinicalPreprocessor:
    continuous_fields: list[str]
    categorical_fields: list[str]

    def __post_init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        for col in self.continuous_fields:
            df[col] = df[col].fillna(df[col].median())
        for col in self.categorical_fields:
            df[col] = df[col].fillna("missing")

        cont = self.scaler.fit_transform(df[self.continuous_fields]) if self.continuous_fields else np.empty((len(df), 0))
        cat = self.encoder.fit_transform(df[self.categorical_fields]) if self.categorical_fields else np.empty((len(df), 0))
        return np.concatenate([cont, cat], axis=1)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        for col in self.continuous_fields:
            df[col] = df[col].fillna(df[col].median())
        for col in self.categorical_fields:
            df[col] = df[col].fillna("missing")

        cont = self.scaler.transform(df[self.continuous_fields]) if self.continuous_fields else np.empty((len(df), 0))
        cat = self.encoder.transform(df[self.categorical_fields]) if self.categorical_fields else np.empty((len(df), 0))
        return np.concatenate([cont, cat], axis=1)


# --------------------------------------------------------------------------------------
# Molecular
# --------------------------------------------------------------------------------------
def select_molecular_features(
    X: np.ndarray,
    y: np.ndarray,
    variance_threshold: float = 0.01,
    top_k: int = 50,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Variance filter followed by mutual-information ranking, keeping the top-K features.

    Returns the reduced feature matrix and the indices (relative to the variance-filtered
    matrix) of the retained features, so the same selection can be re-applied at inference.
    """
    vt = VarianceThreshold(threshold=variance_threshold)
    X_var = vt.fit_transform(X)

    mi = mutual_info_classif(X_var, y, random_state=random_state)
    top_k = min(top_k, X_var.shape[1])
    top_idx = np.argsort(mi)[::-1][:top_k]

    return X_var[:, top_idx], top_idx
