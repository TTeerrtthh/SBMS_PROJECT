from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = PROJECT_ROOT / "dataset" / "raw"

SELECTED_DATA = PROJECT_ROOT / "dataset" / "selected"

PROCESSED_DATA = PROJECT_ROOT / "dataset" / "processed"

RESULTS = PROJECT_ROOT / "results"

MODELS = PROJECT_ROOT / "models"