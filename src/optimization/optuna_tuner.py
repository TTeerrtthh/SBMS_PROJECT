"""
src/optimization/optuna_tuner.py

Hyperparameter tuning using Optuna (Bayesian/TPE search), for:
  1. Plain RandomForestRegressor (baseline comparison model)
  2. AdaptiveWeightedRF (our AHRF core model)

Both use sklearn.model_selection.TimeSeriesSplit for cross-validation on
TRAINING data only -- the test set is never touched here, to avoid leakage
into tuning. TimeSeriesSplit (not ordinary KFold) is required because this
project uses chronological battery-cycle data: hyperparameter search must
never validate on earlier cycles while training on later ones. Each split
is forward/expanding -- fold k's training window is a strict prefix of the
data, and its validation window comes strictly after it in cycle order.

Usage (from a notebook):

    from src.optimization.optuna_tuner import tune_random_forest, tune_weighted_rf

    best_rf_params, study1 = tune_random_forest(X_train, y_train, n_trials=40)
    best_ahrf_params, study2 = tune_weighted_rf(X_train, y_train, n_trials=40)
"""

import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from src.hybrid_model.weighted_random_forest import AdaptiveWeightedRF

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _verify_chronological_splits(n_samples, cv_splitter):
    """
    Verifies that every (train_idx, val_idx) pair produced by cv_splitter
    is genuinely chronological: every validation index occurs strictly
    after every training index (max(train_idx) < min(val_idx)).

    This is a safeguard, not just an assumption -- TimeSeriesSplit already
    guarantees this by construction, but this check makes that guarantee
    explicit and would stop tuning immediately with a clear error if the
    splitting strategy were ever changed in a way that broke it (e.g.
    accidentally swapping back to a shuffled KFold), rather than silently
    letting future information leak into hyperparameter search.

    Checked once per tuning call (splits depend only on n_samples and cv,
    not on the data values, so they are identical across every Optuna trial
    for a given X_train/cv) -- not re-checked inside every trial, to avoid
    redundant work.
    """
    dummy_X = np.arange(n_samples).reshape(-1, 1)
    for train_idx, val_idx in cv_splitter.split(dummy_X):
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        assert max(train_idx) < min(val_idx), (
            "Chronological split check FAILED: training indices must all occur "
            "strictly before validation indices, but got "
            f"max(train_idx)={max(train_idx)} >= min(val_idx)={min(val_idx)}. "
            "Hyperparameter tuning must never validate on earlier cycles while "
            "training on later ones."
        )


def tune_random_forest(X_train, y_train, n_trials=40, cv=5, random_state=42, direction="minimize"):
    """
    Finds good plain-RandomForest hyperparameters. Used for the baseline
    "Random Forest" row in the model comparison table.

    Validation uses TimeSeriesSplit(n_splits=cv) -- a forward/expanding
    chronological split, never shuffled -- instead of ordinary KFold.
    """
    n_samples = len(X_train)
    tscv = TimeSeriesSplit(n_splits=cv)
    _verify_chronological_splits(n_samples, tscv)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        model = RandomForestRegressor(**params, random_state=random_state, n_jobs=-1)
        scores = cross_val_score(
            model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1,
        )
        return -scores.mean()

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Best MAE (cross-validated):", round(study.best_value, 5))
    print("Best parameters:", study.best_params)
    return study.best_params, study


def tune_weighted_rf(X_train, y_train, n_trials=40, cv=5, random_state=42):
    """
    Finds good AdaptiveWeightedRF hyperparameters, including weight_power.

    Validation uses TimeSeriesSplit(n_splits=cv) -- a forward/expanding
    chronological split, never shuffled -- instead of ordinary KFold.
    Uses a manual loop (rather than sklearn's cross_val_score) since
    AdaptiveWeightedRF's OOB-based tree weighting is inspected per-fold
    here for clarity; this preserves the existing AHRF implementation and
    tree-weighting formula unchanged -- only the split strategy is corrected.
    """
    n_samples = len(X_train)
    tscv = TimeSeriesSplit(n_splits=cv)
    _verify_chronological_splits(n_samples, tscv)

    X_arr = np.asarray(X_train)
    y_arr = np.asarray(y_train)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "weight_power": trial.suggest_float("weight_power", 0.0, 4.0),
        }

        fold_maes = []
        for train_idx, test_idx in tscv.split(X_arr):
            # Real, per-fold chronological assertion (not just the one-time
            # pre-check above) -- this runs on every fold, every trial,
            # using the ACTUAL indices tscv produced for this split, so it
            # cannot silently pass due to a stale or mismatched pre-check.
            assert train_idx.max() < test_idx.min(), (
                "Chronological split check FAILED inside AHRF tuning fold: "
                f"train_idx.max()={train_idx.max()} >= test_idx.min()={test_idx.min()}."
            )
            model = AdaptiveWeightedRF(**params, random_state=random_state)
            model.fit(X_arr[train_idx], y_arr[train_idx])
            pred = model.predict(X_arr[test_idx])
            fold_maes.append(mean_absolute_error(y_arr[test_idx], pred))

        return float(np.mean(fold_maes))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Best MAE (cross-validated):", round(study.best_value, 5))
    print("Best parameters:", study.best_params)
    return study.best_params, study
