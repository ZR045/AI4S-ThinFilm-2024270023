from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY
)
from src.model import ThinFilmMLP
from src.utils import set_global_seed, get_device


@torch.no_grad()
def mse_on_arrays(
    model: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> float:
    model.eval()
    total_sq = 0.0
    total_n = 0
    for start in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
        yb = torch.as_tensor(Y[start:start + batch_size], dtype=torch.float32, device=device)
        pred = model(xb)
        total_sq += torch.sum((pred - yb) ** 2).item()
        total_n += yb.numel()
    return total_sq / total_n


def train_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    *,
    train_size: int,
    seed: int = SEED,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> tuple[ThinFilmMLP, pd.DataFrame, dict]:
    """Train one model using the first train_size samples of the fixed training pool."""
    if train_size < 1 or train_size > len(X_train):
        raise ValueError("train_size is outside the training pool")

    set_global_seed(seed)
    device = get_device()

    model = ThinFilmMLP().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()

    Xt = torch.as_tensor(X_train[:train_size], dtype=torch.float32)
    Yt = torch.as_tensor(Y_train[:train_size], dtype=torch.float32)
    dataset = TensorDataset(Xt, Yt)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_sq = 0.0
        train_n = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            train_sq += torch.sum((pred.detach() - yb) ** 2).item()
            train_n += yb.numel()

        train_mse = train_sq / train_n
        val_mse = mse_on_arrays(model, X_val, Y_val, device)

        rows.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})

        if val_mse < best_val:
            best_val = val_mse
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch

    assert best_state is not None
    model.load_state_dict(best_state)

    history = pd.DataFrame(rows)
    meta = {
        "train_size": int(train_size),
        "best_epoch": int(best_epoch),
        "best_val_mse": float(best_val),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "device": str(device),
    }
    return model, history, meta


def save_checkpoint(
    model: ThinFilmMLP,
    path: Path,
    *,
    meta: dict,
) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: Path, device: torch.device | None = None) -> tuple[ThinFilmMLP, dict]:
    if device is None:
        device = get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ThinFilmMLP().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint.get("meta", {})
