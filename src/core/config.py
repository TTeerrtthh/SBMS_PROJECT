from pathlib import Path

# config.py is inside src/core/
# parent        = core
# parent.parent = src
# parent.parent.parent = SBMS_PROJECT

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ==========================
# Dataset Paths
# ==========================

DATASET_DIR = PROJECT_ROOT / "dataset"

RAW_DATA = DATASET_DIR / "raw"

SELECTED_DATA = DATASET_DIR / "selected"

PROCESSED_DATA = DATASET_DIR / "processed"

FEATURE_DATA = DATASET_DIR / "features"

# ==========================
# Results
# ==========================

RESULTS = PROJECT_ROOT / "results"

# ==========================
# Models
# ==========================

MODELS = PROJECT_ROOT / "saved_models"