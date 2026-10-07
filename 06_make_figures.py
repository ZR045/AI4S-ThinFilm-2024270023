from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    DATA_DIR, RESULTS_DIR, FIGURES_DIR, WAVELENGTHS_NM,
    TARGET_WAVELENGTH_NM, LAYER_INDICES
)
from src.data import load_dataset
from src.tmm import reflectance_spectrum
from src.utils import ensure_directories

ensure_directories(FIGURES_DIR)
data = load_dataset(DATA_DIR / "dataset.npz")


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=220, bbox_inches="tight")
    plt.close()


# Figure 1: workflow
fig, ax = plt.subplots(figsize=(12, 2.6))
ax.axis("off")
labels = [
    "Design space\n4 thicknesses\n40–180 nm",
    "TMM physics\n41-point spectrum",
    "Dataset\n5000 samples",
    "MLP surrogate\n4–128–128–64–41",
    "Fast screening\n10,000 candidates",
    "TMM verification\nTop 10 → Top 5",
]
x_positions = np.linspace(0.02, 0.84, len(labels))
for x, label in zip(x_positions, labels):
    ax.add_patch(Rectangle((x, 0.35), 0.14, 0.34, fill=False, linewidth=1.5))
    ax.text(x + 0.07, 0.52, label, ha="center", va="center", fontsize=9)
for i in range(len(labels) - 1):
    ax.annotate("", xy=(x_positions[i + 1], 0.52),
                xytext=(x_positions[i] + 0.14, 0.52),
                arrowprops=dict(arrowstyle="->", lw=1.2))
ax.set_title("AI4S Thin-Film Workflow", fontsize=13)
savefig("workflow.png")


# Figure 2: TMM / dataset schematic
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.axis("off")
layer_names = ["Air", "H", "L", "H", "L", "Glass"]
ys = np.linspace(0.8, 0.2, len(layer_names))
for y, name in zip(ys, layer_names):
    ax.add_patch(Rectangle((0.06, y - 0.045), 0.18, 0.08, fill=False, linewidth=1.3))
    ax.text(0.15, y, name, ha="center", va="center", fontsize=10)
ax.text(0.15, 0.91, "Air / H / L / H / L / Glass", ha="center", fontsize=11)
ax.annotate("", xy=(0.42, 0.5), xytext=(0.26, 0.5), arrowprops=dict(arrowstyle="->"))
ax.text(0.34, 0.56, "d = [d1,d2,d3,d4]", ha="center", fontsize=9)
ax.add_patch(Rectangle((0.42, 0.39), 0.16, 0.22, fill=False, linewidth=1.5))
ax.text(0.50, 0.50, "TMM\n2×2 matrices", ha="center", va="center", fontsize=11)
ax.annotate("", xy=(0.72, 0.5), xytext=(0.58, 0.5), arrowprops=dict(arrowstyle="->"))
ax.add_patch(Rectangle((0.72, 0.39), 0.22, 0.22, fill=False, linewidth=1.5))
ax.text(0.83, 0.50, "R(λ)\n41 wavelengths", ha="center", va="center", fontsize=11)
ax.text(0.50, 0.18, "Dataset matrices: X = 5000 × 4, Y = 5000 × 41",
        ha="center", fontsize=10)
ax.set_title("Physical Model and TMM-Based Data Generation", fontsize=13)
savefig("tmm_dataset.png")


# Figure 3: MLP architecture
fig, ax = plt.subplots(figsize=(10, 3.3))
ax.axis("off")
boxes = [
    ("Input", "4 thicknesses"),
    ("Dense", "128 + ReLU"),
    ("Dense", "128 + ReLU"),
    ("Dense", "64 + ReLU"),
    ("Output", "41 reflectance values"),
]
xs = np.linspace(0.03, 0.81, len(boxes))
for x, (a, b) in zip(xs, boxes):
    ax.add_patch(Rectangle((x, 0.40), 0.16, 0.25, fill=False, linewidth=1.5))
    ax.text(x + 0.08, 0.55, a, ha="center", va="center", fontsize=11)
    ax.text(x + 0.08, 0.46, b, ha="center", va="center", fontsize=8.5)
for i in range(len(boxes) - 1):
    ax.annotate("", xy=(xs[i + 1], 0.525), xytext=(xs[i] + 0.16, 0.525),
                arrowprops=dict(arrowstyle="->", lw=1.2))
ax.set_title("MLP Surrogate Model for Spectrum Prediction", fontsize=13)
savefig("mlp_architecture.png")


# Figure 4: training/validation loss
hist = pd.read_csv(RESULTS_DIR / "history_4000.csv")
plt.figure(figsize=(7, 4.5))
plt.plot(hist["epoch"], hist["train_mse"], label="Training MSE")
plt.plot(hist["epoch"], hist["val_mse"], label="Validation MSE")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.25)
savefig("training_loss.png")


# Figure 5 and 8 use test predictions
with np.load(RESULTS_DIR / "test_predictions_4000.npz") as f:
    Y_true = f["Y_true"]
    Y_pred = f["Y_pred"]
    sample_mse = f["sample_mse"]

order = np.argsort(sample_mse)
rep_idx = [int(order[len(order)//4]), int(order[len(order)//2]), int(order[3*len(order)//4])]

plt.figure(figsize=(8, 5))
for idx in rep_idx:
    plt.plot(WAVELENGTHS_NM, Y_true[idx], linewidth=1.8, label=f"TMM sample {idx}")
    plt.plot(WAVELENGTHS_NM, Y_pred[idx], linestyle="--", linewidth=1.3, label=f"MLP sample {idx}")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.ylim(0, 1)
plt.title("Representative Test Spectra: TMM vs MLP")
plt.legend(ncol=2, fontsize=8)
plt.grid(True, alpha=0.25)
savefig("spectrum_prediction.png")


# Figure 6: effect of data size
dsize = pd.read_csv(RESULTS_DIR / "data_size_metrics.csv")
plt.figure(figsize=(6.5, 4.4))
plt.plot(dsize["train_size"], dsize["mse"], marker="o")
plt.xlabel("Number of Training Samples")
plt.ylabel("Test MSE")
plt.title("Effect of Training-Set Size")
plt.grid(True, alpha=0.25)
savefig("data_size_effect.png")


# Figure 7: best final design
top5 = pd.read_csv(RESULTS_DIR / "top5_candidates.csv")
with np.load(RESULTS_DIR / "design_spectra.npz") as f:
    pred_top10 = f["pred_top10"]
    tmm_top10 = f["tmm_top10"]
    final_idx = f["final_idx"]

best_idx = int(final_idx[0])
plt.figure(figsize=(7, 4.5))
plt.plot(WAVELENGTHS_NM, tmm_top10[best_idx], label="TMM verification", linewidth=2)
plt.plot(WAVELENGTHS_NM, pred_top10[best_idx], "--", label="MLP prediction", linewidth=1.7)
plt.axvline(TARGET_WAVELENGTH_NM, linestyle=":", linewidth=1.5, label=f"Target = {TARGET_WAVELENGTH_NM:.0f} nm")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.ylim(0, 1)
plt.title("Selected Design: MLP Prediction and TMM Verification")
plt.legend()
plt.grid(True, alpha=0.25)
savefig("design_validation.png")


# Figure 8: representative failure case (worst test sample)
worst = int(np.argmax(sample_mse))
plt.figure(figsize=(7, 4.5))
plt.plot(WAVELENGTHS_NM, Y_true[worst], label="TMM", linewidth=2)
plt.plot(WAVELENGTHS_NM, Y_pred[worst], "--", label="MLP", linewidth=1.7)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.ylim(0, 1)
plt.title(f"Representative Failure Case (test sample {worst})")
plt.legend()
plt.grid(True, alpha=0.25)
savefig("failure_case.png")

print("Saved figures to:", FIGURES_DIR)
