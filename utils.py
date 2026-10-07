from __future__ import annotations

import random
from pathlib import Path
import numpy as np
import torch


def ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # The current model uses deterministic CPU/GPU operations.
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
