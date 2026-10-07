from __future__ import annotations

import numpy as np
import torch

from src.utils import get_device


@torch.no_grad()
def predict(model: torch.nn.Module, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    device = get_device()
    model = model.to(device)
    model.eval()
    outputs = []
    for start in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
        outputs.append(model(xb).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    return {
        "mse": float(np.mean(error ** 2)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def per_sample_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean((y_pred - y_true) ** 2, axis=1)
