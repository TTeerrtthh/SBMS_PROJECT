"""
src/core/config.py

SINGLE SOURCE OF TRUTH for every folder path in the project.
Every notebook and every module imports paths from here. Never hardcode
a path anywhere else — if a folder ever needs to move, this is the only
file that changes.
"""

from pathlib import Path

# Project root = two levels up from this file (src/core/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- Dataset paths ----
DATASET_DIR = PROJECT_ROOT / "dataset"

RAW_DATA = DATASET_DIR / "raw"
NASA_ORIGINAL = RAW_DATA / "nasa_original"      # untouched original NASA files, kept as reference
METADATA_DIR = RAW_DATA / "metadata"            # metadata.csv lives here

PROCESSED_DATA = DATASET_DIR / "processed"      # PROCESSED_DATA/<battery_id>/{charge,discharge,impedance}/*.csv
FEATURE_DATA = DATASET_DIR / "features"         # engineered feature CSVs + label CSVs land here

# ---- Output paths ----
MODELS = PROJECT_ROOT / "models"
RESULTS = PROJECT_ROOT / "results"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

# ---- Battery constants ----
# NASA B0005/B0006/B0007/B0018 cells are rated at 2.0 Ah.
RATED_CAPACITY_AH = 2.0

# SOH threshold (%) that defines "End of Life" for RUL calculation.
# 80% is the standard convention in battery-health literature.
EOL_SOH_THRESHOLD = 80.0

# Which battery(s) this project currently processes.
# Add more IDs here later (e.g. ["B0005", "B0006"]) without changing
# any other file — every module reads this list.
BATTERY_IDS = ["B0005"]


def ensure_dirs():
    """Creates every folder above if it doesn't already exist.
    Safe to call at the top of any notebook."""
    for path in [
        RAW_DATA, NASA_ORIGINAL, METADATA_DIR,
        PROCESSED_DATA, FEATURE_DATA,
        MODELS, RESULTS, REPORTS, FIGURES,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    for battery_id in BATTERY_IDS:
        for experiment in ["charge", "discharge", "impedance"]:
            (PROCESSED_DATA / battery_id / experiment).mkdir(parents=True, exist_ok=True)
