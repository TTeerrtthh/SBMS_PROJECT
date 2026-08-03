"""
src/optimization/optuna_tuner.py

Hyperparameter tuning using Optuna (Bayesian/TPE search), for:
  1. Plain RandomForestRegressor (baseline comparison model)
  2. AdaptiveWeightedRF (our AHRF core model)

Both use standard k-fold cross-validation on TRAINING data only -- the
test set is never touched here, to avoid leakage into tuning.

Usage (from a notebook):

    from src.optimization.optuna_tuner import tune_random_forest, tune_weighted_rf

    best_rf_params, study1 = tune_random_forest(X_train, y_train, n_trials=40)
    best_ahrf_params, study2 = tune_weighted_rf(X_train, y_train, n_trials=40)
"""

import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error

from src.hybrid_model.weighted_random_forest import AdaptiveWeightedRF

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_random_forest(X_train, y_train, n_trials=40, cv=5, random_state=42, direction="minimize"):
    """
    Finds good plain-RandomForest hyperparameters. Used for the baseline
    "Random Forest" row in the model comparison table.
    """

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
            model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1,
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

    Uses manual K-fold cross-validation (rather than sklearn's cross_val_score,
    since AdaptiveWeightedRF's .fit(X, y) internally uses OOB scoring, which
    is compatible with cross_val_score, but we keep this explicit for clarity
    and so it's easy to inspect the per-fold OOB weighting behavior if needed).
    """

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "weight_power": trial.suggest_float("weight_power", 0.0, 4.0),
        }

        kf = KFold(n_splits=cv, shuffle=False)  # shuffle=False: keep folds in
        # original (chronological) row order, consistent with the rest of
        # this leakage-safe project.
        fold_maes = []

        X_arr = np.asarray(X_train)
        y_arr = np.asarray(y_train)

        for train_idx, test_idx in kf.split(X_arr):
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
