from __future__ import annotations

import numpy as np

from config import N_AIR, N_SUBSTRATE, LAYER_INDICES, WAVELENGTHS_NM


def reflectance_batch(
    thicknesses_nm: np.ndarray,
    wavelengths_nm: np.ndarray = WAVELENGTHS_NM,
    layer_indices: np.ndarray = LAYER_INDICES,
    n_incident: float = N_AIR,
    n_substrate: float = N_SUBSTRATE,
) -> np.ndarray:
    """
    Normal-incidence characteristic-matrix TMM for lossless, non-dispersive layers.

    Parameters
    ----------
    thicknesses_nm : (N, L) or (L,) array
        Layer thicknesses in nm.
    wavelengths_nm : (W,) array
        Wavelengths in nm.
    layer_indices : (L,) array
        Refractive indices from incident side to substrate side.

    Returns
    -------
    R : (N, W) array
        Power reflectance.
    """
    d = np.asarray(thicknesses_nm, dtype=np.float64)
    if d.ndim == 1:
        d = d[None, :]
    if d.ndim != 2:
        raise ValueError("thicknesses_nm must be a 1D or 2D array")

    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    indices = np.asarray(layer_indices, dtype=np.float64)

    if d.shape[1] != indices.size:
        raise ValueError("Number of thickness values must match number of layers")
    if np.any(wavelengths <= 0):
        raise ValueError("Wavelengths must be positive")
    if np.any(indices <= 0) or n_incident <= 0 or n_substrate <= 0:
        raise ValueError("Refractive indices must be positive")

    n_samples = d.shape[0]
    n_waves = wavelengths.size

    # Total characteristic matrix starts as identity.
    A = np.ones((n_samples, n_waves), dtype=np.complex128)
    B = np.zeros_like(A)
    C = np.zeros_like(A)
    D = np.ones_like(A)

    for layer, n_layer in enumerate(indices):
        delta = 2.0 * np.pi * n_layer * d[:, layer, None] / wavelengths[None, :]
        c = np.cos(delta)
        s = np.sin(delta)

        a = c
        b = 1j * s / n_layer
        cmat = 1j * n_layer * s
        dmat = c

        new_A = A * a + B * cmat
        new_B = A * b + B * dmat
        new_C = C * a + D * cmat
        new_D = C * b + D * dmat
        A, B, C, D = new_A, new_B, new_C, new_D

    y_in = (C + D * n_substrate) / (A + B * n_substrate)
    r = (n_incident - y_in) / (n_incident + y_in)
    R = np.abs(r) ** 2

    # Clip only tiny floating-point overshoots for a passive lossless system.
    return np.clip(R.real, 0.0, 1.0)


def reflectance_spectrum(
    thickness_nm: np.ndarray,
    wavelengths_nm: np.ndarray = WAVELENGTHS_NM,
) -> np.ndarray:
    """Convenience wrapper returning a 1D spectrum for one 4-layer design."""
    return reflectance_batch(thickness_nm, wavelengths_nm=wavelengths_nm)[0]
