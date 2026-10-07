from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import generate_dataset


def test_dataset_seed_reproducibility():
    a = generate_dataset(270023)
    b = generate_dataset(270023)
    assert np.array_equal(a["X_train"], b["X_train"])
    assert np.array_equal(a["Y_test"], b["Y_test"])


def test_different_seed_changes_dataset():
    a = generate_dataset(270023)
    b = generate_dataset(270024)
    assert not np.array_equal(a["X_train"], b["X_train"])
