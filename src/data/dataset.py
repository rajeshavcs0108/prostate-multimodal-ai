"""PyTorch Dataset/DataLoader for the multimodal (imaging, clinical, molecular) cohort."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class MultimodalProstateDataset(Dataset):
    """Expects pre-extracted, pre-processed feature arrays (see `src/data/preprocessing.py`
    and `scripts/prepare_data.py`), saved as .npy files:

        imaging.npy    -> (N, D_img)   radiomic feature vectors R
        clinical.npy   -> (N, D_clin)  normalized + one-hot encoded clinical vectors
        molecular.npy  -> (N, D_mol)   top-K selected molecular features
        labels.npy     -> (N,)         binary aggressiveness label (0 = non-aggressive, 1 = aggressive)
    """

    def __init__(self, imaging: np.ndarray, clinical: np.ndarray, molecular: np.ndarray, labels: np.ndarray):
        assert len(imaging) == len(clinical) == len(molecular) == len(labels)
        self.imaging = torch.as_tensor(imaging, dtype=torch.float32)
        self.clinical = torch.as_tensor(clinical, dtype=torch.float32)
        self.molecular = torch.as_tensor(molecular, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "imaging": self.imaging[idx],
            "clinical": self.clinical[idx],
            "molecular": self.molecular[idx],
            "label": self.labels[idx],
        }

    @classmethod
    def from_dir(cls, processed_dir: str) -> "MultimodalProstateDataset":
        import os

        imaging = np.load(os.path.join(processed_dir, "imaging.npy"))
        clinical = np.load(os.path.join(processed_dir, "clinical.npy"))
        molecular = np.load(os.path.join(processed_dir, "molecular.npy"))
        labels = np.load(os.path.join(processed_dir, "labels.npy"))
        return cls(imaging, clinical, molecular, labels)
