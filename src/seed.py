"""Deterministic seeding utilities.

Fixing seeds across Python's `random`, NumPy, and PyTorch (CPU + CUDA) so that the
five independent runs referenced in the manuscript ("five independent runs of the
experiment") are individually reproducible.
"""
import os
import random

import numpy as np
import torch

RNG_SEED = 42


def set_seed(seed: int = RNG_SEED, deterministic: bool = True) -> None:
    """Seed all RNG sources and (optionally) force deterministic CUDA kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some ops may not have deterministic kernels on all hardware; don't hard-fail.
            pass
