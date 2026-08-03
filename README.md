<<<<<<< HEAD
# SBMS Project — Adaptive Hybrid Random Forest (AHRF) for Battery Health Prediction

Predicts **SOC**, **SOH**, and **RUL** for NASA lithium-ion battery B0005 using
AHRF — a Random Forest where each tree's vote is weighted by its own
out-of-bag (OOB) accuracy, instead of every tree voting equally.

## Locked Project Structure — do not reorganize

```
SBMS_PROJECT/
├── dataset/
│   ├── raw/
│   │   ├── nasa_original/     # (optional) untouched original NASA files, for reference
│   │   └── metadata/
│   │       └── metadata.csv   # PUT YOUR METADATA CSV HERE
│   ├── processed/
│   │   └── B0005/
│   │       ├── charge/        # PUT YOUR SEGREGATED CHARGE CSVs HERE
│   │       ├── discharge/     # PUT YOUR SEGREGATED DISCHARGE CSVs HERE
│   │       └── impedance/     # PUT YOUR SEGREGATED IMPEDANCE CSVs HERE
│   └── features/              # engineered feature + label CSVs land here (auto-generated)
├── models/                    # saved .pkl models (auto-generated)
├── results/                   # metrics CSVs, comparison tables (auto-generated)
├── reports/
│   └── figures/                # saved plots for your report (auto-generated)
├── notebooks/                 # 01 through 08, run in order -- see list below
└── src/                       # all reusable logic — notebooks call these, never duplicate logic
    ├── core/
    │   ├── config.py          # every file path — single source of truth
    │   └── data_loader.py     # loads metadata + processed cycle files
    ├── preprocessing/
    │   └── cleaner.py         # cleans cycle data, validates folder structure
    ├── feature_engineering/
    │   └── extractor.py       # per-cycle statistical features
    ├── labels/
    │   └── ground_truth.py    # SOC / SOH / RUL label generation
    ├── optimization/
    │   └── optuna_tuner.py    # hyperparameter tuning (RF baseline + AHRF)
    ├── hybrid_model/
    │   └── weighted_random_forest.py   # AdaptiveWeightedRF (AHRF core)
    ├── explainability/
    │   └── shap_utils.py      # SHAP plots
    └── decision_engine/
        └── rules.py           # SOC/SOH/RUL -> maintenance recommendation
```

**Rule: notebooks only orchestrate (load data → call a function from `src/` →
show/save output). All real logic lives in `src/`.** This keeps every
notebook short, readable, and bug-fixes only need to happen in one place.

## Notebooks — run in this exact order

| # | Notebook | What it does |
|---|---|---|
| 01 | `data_understanding.ipynb` | Validates folder structure, inspects metadata, checks filename = chronological order |
| 02 | `preprocessing.ipynb` | Cleans every discharge cycle (missing values, dtypes) |
| 03 | `feature_engineering.ipynb` | Extracts ~60-70 statistical features per cycle |
| 04 | `ground_truth_labels.ipynb` | Generates SOC, SOH, RUL labels |
| 05 | `model_training.ipynb` | Trains + tunes AHRF vs. baseline RF for SOC, SOH, RUL |
| 06 | `model_comparison.ipynb` | Benchmarks AHRF against 5 more models (Linear, Poly, DT, RF, GB) |
| 07 | `explainability_shap.ipynb` | SHAP global + local explanations |
| 08 | `decision_engine_summary.ipynb` | Maintenance recommendations + final results compilation |

Every notebook has been executed end-to-end against a synthetic dataset matching this exact structure to confirm there are no errors, before you plug in your real sensor data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Before running any notebook

1. Place `metadata.csv` in `dataset/raw/metadata/`.
2. Confirm your already-segregated cycle CSVs are in
   `dataset/processed/B0005/{charge,discharge,impedance}/`.
3. Open `src/core/config.py` and check `RATED_CAPACITY_AH` and
   `EOL_SOH_THRESHOLD` match your intended convention (defaults: 2.0 Ah, 80%).
4. Run `notebooks/01_data_understanding.ipynb` first — it calls
   `validate_processed_structure()`, which checks your folders are correctly
   populated before anything else runs.

## AHRF, in one sentence

A Random Forest whose final prediction is a weighted (not simple) average of
its trees, where each tree's weight comes from its own out-of-bag accuracy —
mathematically reduces to plain Random Forest when `weight_power=0`, and
requires zero extra data (no validation split needed) since it reuses OOB
samples that bootstrapping already sets aside.

## Tested

Every module in `src/` has been run end-to-end against a synthetic dataset
matching this exact folder/column structure (120 discharge cycles, degrading
capacity, matching metadata) to confirm the full pipeline runs without
errors before you plug in your real data.
=======
# SBMS
>>>>>>> 321d3fa708c1f9828179672d4064df9bdd6dbbee
