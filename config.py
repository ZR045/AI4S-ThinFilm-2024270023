"""Central configuration for the AI4S thin-film project."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

STUDENT_ID = "2024270023"
LAST_TWO = int(STUDENT_ID[-2:])
SEED = int(STUDENT_ID[-6:])
DESIGN_SEED = SEED + 1
TARGET_WAVELENGTH_NM = 450 + 10 * (LAST_TWO % 31)

N_AIR = 1.00
N_H = 2.30
N_L = 1.45
N_SUBSTRATE = 1.52
LAYER_INDICES = np.array([N_H, N_L, N_H, N_L], dtype=np.float64)

THICKNESS_MIN_NM = 40.0
THICKNESS_MAX_NM = 180.0

WAVELENGTH_MIN_NM = 400.0
WAVELENGTH_MAX_NM = 800.0
WAVELENGTH_STEP_NM = 10.0
WAVELENGTHS_NM = np.arange(
    WAVELENGTH_MIN_NM,
    WAVELENGTH_MAX_NM + WAVELENGTH_STEP_NM,
    WAVELENGTH_STEP_NM,
    dtype=np.float64,
)

N_TOTAL = 5000
N_TRAIN = 4000
N_VAL = 500
N_TEST = 500

ARCHITECTURE = (4, 128, 128, 64, 41)
TRAIN_SIZES = (500, 1000, 2000, 4000)

BATCH_SIZE = 128
EPOCHS = 700
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 0.0

N_DESIGN_CANDIDATES = 10000
TOP_K_MLP = 10
FINAL_TOP_K = 5

# The assignment says to rank candidates by reflectance at lambda_target
# but does not explicitly state ascending/descending. This project defaults
# to maximizing reflectance. Change to "minimize" if your instructor says so.
DESIGN_OBJECTIVE = "maximize"
