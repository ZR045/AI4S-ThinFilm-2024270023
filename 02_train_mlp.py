from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR, MODEL_DIR, RESULTS_DIR, N_TRAIN
from src.data import load_dataset
from src.training import train_model, save_checkpoint
from src.utils import ensure_directories

ensure_directories(MODEL_DIR, RESULTS_DIR)
data = load_dataset(DATA_DIR / "dataset.npz")

model, history, meta = train_model(
    data["X_train"], data["Y_train"],
    data["X_val"], data["Y_val"],
    train_size=N_TRAIN,
)

save_checkpoint(model, MODEL_DIR / "mlp_4000.pt", meta=meta)
history.to_csv(RESULTS_DIR / "history_4000.csv", index=False)

print("Training complete.")
print(meta)
