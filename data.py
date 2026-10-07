from __future__ import annotations

from pathlib import Path
import numpy as np

from config import (
    SEED, N_TOTAL, N_TRAIN, N_VAL, N_TEST,
    THICKNESS_MIN_NM, THICKNESS_MAX_NM, WAVELENGTHS_NM
)
from src.tmm import reflectance_batch


def generate_dataset(seed: int = SEED) -> dict[str, np.ndarray]:
    """Generate 5000 thickness-spectrum pairs and make the fixed 4000/500/500 split."""
    rng = np.random.default_rng(seed)

    thicknesses = rng.uniform(
        THICKNESS_MIN_NM,
        THICKNESS_MAX_NM,
        size=(N_TOTAL, 4),
    )
    spectra = reflectance_batch(thicknesses)

    # One fixed seed-controlled permutation, then contiguous fixed splits.
    order = rng.permutation(N_TOTAL)
    thicknesses = thicknesses[order]
    spectra = spectra[order]

    i1 = N_TRAIN
    i2 = N_TRAIN + N_VAL
    assert i2 + N_TEST == N_TOTAL

    return {
        "wavelengths_nm": WAVELENGTHS_NM.copy(),
        "X_train": thicknesses[:i1],
        "Y_train": spectra[:i1],
        "X_val": thicknesses[i1:i2],
        "Y_val": spectra[i1:i2],
        "X_test": thicknesses[i2:],
        "Y_test": spectra[i2:],
    }


def save_dataset(path: Path, seed: int = SEED) -> dict[str, np.ndarray]:
    data = generate_dataset(seed)
    np.savez_compressed(path, **data)
    return data


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {key: f[key] for key in f.files}
