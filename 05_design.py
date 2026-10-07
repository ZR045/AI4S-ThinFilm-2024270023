from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    MODEL_DIR, RESULTS_DIR, TARGET_WAVELENGTH_NM, DESIGN_OBJECTIVE
)
from src.design import verify_and_rank_top
from src.training import load_checkpoint
from src.utils import ensure_directories

ensure_directories(RESULTS_DIR)
model, _ = load_checkpoint(MODEL_DIR / "mlp_4000.pt")
out = verify_and_rank_top(model)

rows = []
for i in range(len(out["X_top10"])):
    x = out["X_top10"][i]
    rows.append({
        "mlp_rank": i + 1,
        "d1_nm": x[0],
        "d2_nm": x[1],
        "d3_nm": x[2],
        "d4_nm": x[3],
        "mlp_R_target": out["mlp_scores_top10"][i],
        "tmm_R_target": out["tmm_scores_top10"][i],
        "abs_error_target": abs(out["mlp_scores_top10"][i] - out["tmm_scores_top10"][i]),
    })

top10 = pd.DataFrame(rows)
top10.to_csv(RESULTS_DIR / "top10_candidates.csv", index=False)

final_rows = []
for final_rank, idx in enumerate(out["final_idx"], start=1):
    row = rows[int(idx)].copy()
    row["final_tmm_rank"] = final_rank
    final_rows.append(row)

top5 = pd.DataFrame(final_rows)[
    ["final_tmm_rank", "mlp_rank", "d1_nm", "d2_nm", "d3_nm", "d4_nm",
     "mlp_R_target", "tmm_R_target", "abs_error_target"]
]
top5.to_csv(RESULTS_DIR / "top5_candidates.csv", index=False)

np.savez_compressed(
    RESULTS_DIR / "design_spectra.npz",
    X_top10=out["X_top10"],
    pred_top10=out["pred_top10"],
    tmm_top10=out["tmm_top10"],
    final_idx=out["final_idx"],
)

print(f"Objective: {DESIGN_OBJECTIVE} R({TARGET_WAVELENGTH_NM:.0f} nm)")
print(top5.to_string(index=False))
