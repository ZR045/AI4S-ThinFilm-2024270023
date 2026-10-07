from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR, SEED
from src.data import save_dataset
from src.utils import ensure_directories

ensure_directories(DATA_DIR)
path = DATA_DIR / "dataset.npz"
data = save_dataset(path, seed=SEED)

print(f"Saved: {path}")
for key in ["X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test"]:
    print(f"{key}: {data[key].shape}")
