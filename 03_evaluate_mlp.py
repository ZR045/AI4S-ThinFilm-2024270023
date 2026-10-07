from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR, MODEL_DIR, RESULTS_DIR
from src.data import load_dataset
from src.evaluation import predict, regression_metrics, per_sample_mse
from src.training import load_checkpoint
from src.utils import ensure_directories

ensure_directories(RESULTS_DIR)
data = load_dataset(DATA_DIR / "dataset.npz")
model, meta = load_checkpoint(MODEL_DIR / "mlp_4000.pt")

pred = predict(model, data["X_test"])
metrics = regression_metrics(data["Y_test"], pred)
sample_mse = per_sample_mse(data["Y_test"], pred)

metrics_row = {"train_size": 4000, **metrics}
pd.DataFrame([metrics_row]).to_csv(RESULTS_DIR / "metrics_4000.csv", index=False)

np.savez_compressed(
    RESULTS_DIR / "test_predictions_4000.npz",
    X_test=data["X_test"],
    Y_true=data["Y_test"],
    Y_pred=pred,
    sample_mse=sample_mse,
    wavelengths_nm=data["wavelengths_nm"],
)

worst = int(np.argmax(sample_mse))
with open(RESULTS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "model_meta": meta,
            "test_metrics": metrics,
            "worst_test_index": worst,
            "worst_sample_mse": float(sample_mse[worst]),
        },
        f, indent=2
    )

print("Test metrics:", metrics)
print("Worst test sample:", worst, "MSE =", float(sample_mse[worst]))
