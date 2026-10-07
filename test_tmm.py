from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tmm import reflectance_batch, reflectance_spectrum


def test_zero_thickness_reduces_to_air_glass_interface():
    wavelengths = np.array([400.0, 550.0, 800.0])
    R = reflectance_spectrum(np.zeros(4), wavelengths)
    expected = ((1.0 - 1.52) / (1.0 + 1.52)) ** 2
    assert np.allclose(R, expected, atol=1e-12)


def test_boundary_thicknesses_are_physical():
    X = np.array([
        [40.0, 40.0, 40.0, 40.0],
        [180.0, 180.0, 180.0, 180.0],
        [40.0, 180.0, 40.0, 180.0],
    ])
    R = reflectance_batch(X)
    assert R.shape == (3, 41)
    assert np.all(np.isfinite(R))
    assert np.all((R >= 0.0) & (R <= 1.0))


def test_single_and_batch_interfaces_match():
    x = np.array([100.0, 110.0, 120.0, 130.0])
    r1 = reflectance_spectrum(x)
    r2 = reflectance_batch(x[None, :])[0]
    assert np.allclose(r1, r2)
