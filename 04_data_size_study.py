from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR, MODEL_DIR, RESULTS_DIR, TRAIN_SIZES
from src.data import load_dataset
from src.training import train_model, save_checkpoint, load_checkpoint
from src.evaluation import predict, regression_metrics
from src.utils import ensure_directories

ensure_directories(MODEL_DIR, RESULTS_DIR)
data = load_dataset(DATA_DIR / "dataset.npz")
rows = []

for n in TRAIN_SIZES:
    model_path = MODEL_DIR / f"mlp_{n}.pt"
    history_path = RESULTS_DIR / f"history_{n}.csv"

    if n == 4000 and model_path.exists() and history_path.exists():
        model, meta = load_checkpoint(model_path)
        history = pd.read_csv(history_path)
    else:
        model, history, meta = train_model(
            data["X_train"], data["Y_train"],
            data["X_val"], data["Y_val"],
            train_size=n,
        )
        save_checkpoint(model, model_path, meta=meta)
        history.to_csv(history_path, index=False)

    pred = predict(model, data["X_test"])
    metrics = regression_metrics(data["Y_test"], pred)
    rows.append({
        "train_size": n,
        "best_epoch": meta["best_epoch"],
        "best_val_mse": meta["best_val_mse"],
        **metrics,
    })
    print(n, metrics)

df = pd.DataFrame(rows).sort_values("train_size")
df.to_csv(RESULTS_DIR / "data_size_metrics.csv", index=False)
print("Saved:", RESULTS_DIR / "data_size_metrics.csv")
