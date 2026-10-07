from __future__ import annotations

import numpy as np

from config import (
    DESIGN_SEED, N_DESIGN_CANDIDATES, THICKNESS_MIN_NM, THICKNESS_MAX_NM,
    WAVELENGTHS_NM, TARGET_WAVELENGTH_NM, TOP_K_MLP, FINAL_TOP_K, DESIGN_OBJECTIVE
)
from src.evaluation import predict
from src.tmm import reflectance_batch


def target_index() -> int:
    idx = np.where(np.isclose(WAVELENGTHS_NM, TARGET_WAVELENGTH_NM))[0]
    if idx.size != 1:
        raise RuntimeError("Target wavelength is not on the configured wavelength grid")
    return int(idx[0])


def screen_candidates(model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(DESIGN_SEED)
    X = rng.uniform(
        THICKNESS_MIN_NM,
        THICKNESS_MAX_NM,
        size=(N_DESIGN_CANDIDATES, 4),
    )
    pred = predict(model, X)
    idx = target_index()
    scores = pred[:, idx]

    if DESIGN_OBJECTIVE == "maximize":
        order = np.argsort(scores)[::-1]
    elif DESIGN_OBJECTIVE == "minimize":
        order = np.argsort(scores)
    else:
        raise ValueError('DESIGN_OBJECTIVE must be "maximize" or "minimize"')

    top_idx = order[:TOP_K_MLP]
    return X[top_idx], pred[top_idx], scores[top_idx]


def verify_and_rank_top(model):
    X_top, pred_top, mlp_scores = screen_candidates(model)
    tmm_top = reflectance_batch(X_top)
    idx = target_index()
    tmm_scores = tmm_top[:, idx]

    if DESIGN_OBJECTIVE == "maximize":
        verify_order = np.argsort(tmm_scores)[::-1]
    else:
        verify_order = np.argsort(tmm_scores)

    final_idx = verify_order[:FINAL_TOP_K]
    return {
        "X_top10": X_top,
        "pred_top10": pred_top,
        "tmm_top10": tmm_top,
        "mlp_scores_top10": mlp_scores,
        "tmm_scores_top10": tmm_scores,
        "final_idx": final_idx,
    }
