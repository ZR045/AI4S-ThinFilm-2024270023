from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "01_generate_dataset.py",
    "02_train_mlp.py",
    "03_evaluate_mlp.py",
    "04_data_size_study.py",
    "05_design.py",
    "06_make_figures.py",
]

for script in SCRIPTS:
    path = ROOT / "scripts" / script
    print(f"\n=== Running {script} ===", flush=True)
    subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)

print("\nAll experiments completed successfully.")
