from pathlib import Path

# config.py is inside src/core/
# parent      = core
# parent.parent = src
# parent.parent.parent = SBMS_PROJECT

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_DATA = PROJECT_ROOT / "dataset" / "raw"

SELECTED_DATA = PROJECT_ROOT / "dataset" / "selected"

PROCESSED_DATA = PROJECT_ROOT / "dataset" / "processed"

RESULTS = PROJECT_ROOT / "results"

MODELS = PROJECT_ROOT / "models"