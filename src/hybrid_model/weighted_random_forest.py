"""
src/hybrid_model/weighted_random_forest.py

AdaptiveWeightedRF: a Random Forest whose final prediction is a WEIGHTED
average of its trees, instead of the equal-weight average plain
RandomForestRegressor uses. Two research versions of the weighting
mechanism are implemented, selected via `weighting_method`:

  "oob"      (AHRF-v1, default) -- OOB-error-based adaptive tree
             weighting. Each tree's weight is based on its own
             out-of-bag (OOB) accuracy: the tree's error on the training
             rows IT never saw during its own bootstrap sampling. Cheap
             (reuses bootstrapping's "free" held-out rows) but OOB rows
             are a random subset of the training period, not
             specifically the LATER cycles.

  "temporal" (AHRF-v2) -- Temporal-adaptive tree weighting using
             genuinely out-of-sample forward validation from
             independently trained fold-specific forests. For each
             TimeSeriesSplit fold, a COMPLETELY SEPARATE
             RandomForestRegressor is trained using only that fold's
             training rows, then evaluated only on that fold's
             validation rows (later cycles the fold forest never saw).
             Each tree SLOT's (index i, sharing the same deterministic
             random seed across folds and across the final forest, since
             all forests use the same n_estimators/random_state) error is
             averaged across folds, producing a genuinely out-of-sample
             temporal reliability estimate per slot. That estimate is
             then TRANSFERRED to the corresponding tree in the final
             forest (fit separately, once, on the complete X_train).

             IMPORTANT, PRECISE CLAIM: the fold-specific tree at index i
             and the final full-data tree at index i are NOT the same
             fitted tree object -- they were grown from different
             (differently-sized) training data. This class does NOT
             claim "the final full-data tree was evaluated on unseen
             validation data", because that would be false. What it
             actually does: temporal reliability of corresponding
             deterministic tree slots was estimated using independently
             trained forward-validation forests, then transferred to the
             final full-data forest, on the reasoning that a given
             seed slot's tree-growing behavior (which features/splits it
             tends to prefer) is a reasonably stable property of that
             slot across similarly-configured forests trained on
             overlapping data from the same underlying process.

  Default remains "oob", so existing code (including the already-locked
  Notebook 05) is unaffected unless it explicitly opts into "temporal".

  AHRF-v2 is an EXPERIMENTAL RESEARCH VARIANT. It is not assumed or
  claimed to be superior to AHRF-v1 -- that must be established
  empirically by Notebook 05 (or a successor) experiments, not asserted
  here.

IMPLEMENTATION NOTE (OOB path): earlier versions of this file used
sklearn's private `sklearn.ensemble._forest._generate_sample_indices()`
to reconstruct each tree's bootstrap sample. That function's signature
changed between sklearn versions (some require an extra `sample_weight`
argument), which caused a TypeError on some setups. This version
reimplements the same bootstrap logic directly using
`sklearn.utils.check_random_state`, a stable, public utility -- so this
class no longer depends on any sklearn internal/private API.

Usage (from a notebook):

    from src.hybrid_model.weighted_random_forest import AdaptiveWeightedRF

    # AHRF-v1 (default, unchanged behavior)
    model_v1 = AdaptiveWeightedRF(**best_params_from_optuna)
    model_v1.fit(X_train, y_train)
    y_pred = model_v1.predict(X_test)
    print(model_v1.tree_oob_errors_)

    # AHRF-v2 (opt-in, experimental)
    model_v2 = AdaptiveWeightedRF(**best_params_from_optuna, weighting_method="temporal")
    model_v2.fit(X_train, y_train)
    y_pred = model_v2.predict(X_test)
    print(model_v2.tree_temporal_errors_)
    print(model_v2.temporal_fold_tree_errors_)   # shape (n_estimators, n_temporal_splits)
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import check_random_state
from sklearn.metrics import mean_absolute_error


def _bootstrap_sample_indices(random_state, n_samples):
    """
    Reconstructs the same bootstrap sample indices sklearn's Random Forest
    uses internally for a tree with the given random_state, WITHOUT
    depending on sklearn's private _generate_sample_indices function.
    This is exactly what that private function does internally (sampling
    n_samples indices, with replacement, from range(n_samples)), just
    implemented directly using the public check_random_state utility so
    it's stable across sklearn versions.
    """
    random_instance = check_random_state(random_state)
    return random_instance.randint(0, n_samples, n_samples)


class AdaptiveWeightedRF:
    """
    A Random Forest whose final prediction is a WEIGHTED average of its
    trees' predictions. See module docstring for the two supported
    weighting mechanisms ("oob" = AHRF-v1, "temporal" = AHRF-v2).

    Plain RandomForestRegressor prediction:
        y_pred = mean(tree_1(x), tree_2(x), ..., tree_N(x))

    AdaptiveWeightedRF prediction (either weighting_method):
        y_pred = sum(w_i * tree_i(x))  where sum(w_i) = 1
        and w_i is higher for trees with lower error (OOB error for
        "oob", mean cross-fold out-of-sample temporal error for
        "temporal", transferred by tree-slot index to the final forest).
    """

    def __init__(self, n_estimators=200, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", random_state=42,
                 weight_power=1.0, weighting_method="oob", n_temporal_splits=3):
        """
        weight_power : controls how aggressively good trees are favored,
            for EITHER weighting_method.
            0.0 -> all trees weighted equally (mathematically reduces to plain RF).
            1.0 -> weight directly proportional to inverse error (default).
            Higher values (2.0-4.0) favor the best trees more strongly.
            Tune this alongside the usual RF parameters via Optuna.

        weighting_method : "oob" (default, AHRF-v1) or "temporal" (AHRF-v2,
            experimental). "oob" is unchanged from the original
            implementation and is the default, so existing code is
            unaffected unless it opts into "temporal" explicitly. No
            additional parameter is required to use AHRF-v1.

        n_temporal_splits : number of TimeSeriesSplit folds used ONLY when
            weighting_method="temporal" (ignored for "oob"). For each
            fold, a completely separate RandomForestRegressor is trained
            on that fold's training rows only, and evaluated on that
            fold's validation rows only -- both carved out of the
            training data supplied to .fit(). The final test set is
            never touched by this class.

        Note: bootstrap=True is required (and fixed) for OOB weighting to
        be possible -- this class does not expose a bootstrap parameter.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.weight_power = weight_power
        self.weighting_method = weighting_method
        self.n_temporal_splits = n_temporal_splits

        self.forest_ = None
        self.tree_weights_ = None
        self.tree_oob_errors_ = None
        self.tree_temporal_errors_ = None
        self.temporal_fold_tree_errors_ = None
        self.temporal_fold_scores_ = None

    def _make_forest(self):
        """
        Builds a fresh, UNFIT RandomForestRegressor using this instance's
        configured hyperparameters. Used identically for every
        fold-specific forest AND the final forest, so n_estimators (and
        every other hyperparameter) can never silently drift/mismatch
        between them -- they are always constructed from this one place.
        """
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=True,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def _compute_oob_errors(self, X_arr, y_arr):
        """AHRF-v1: per-tree OOB MAE, from the already-fitted self.forest_.
        Unchanged from the original implementation."""
        n_samples = X_arr.shape[0]
        errors = np.full(self.n_estimators, np.nan)

        for i, tree in enumerate(self.forest_.estimators_):
            # Reconstruct which training rows THIS tree's bootstrap sample
            # included, using its stored random_state.
            sample_indices = _bootstrap_sample_indices(tree.random_state, n_samples)
            in_bag = np.zeros(n_samples, dtype=bool)
            in_bag[sample_indices] = True
            oob_mask = ~in_bag

            if oob_mask.sum() == 0:
                errors[i] = np.nan
                continue

            oob_pred = tree.predict(X_arr[oob_mask])
            errors[i] = mean_absolute_error(y_arr[oob_mask], oob_pred)

        if np.isnan(errors).any():
            fallback = np.nanmean(errors)
            errors = np.where(np.isnan(errors), fallback, errors)

        return errors

    def _compute_temporal_errors(self, X_arr, y_arr):
        """
        AHRF-v2: genuine out-of-sample temporal reliability per tree slot.

        For each TimeSeriesSplit fold (train_idx strictly before val_idx,
        asserted): trains a COMPLETELY SEPARATE RandomForestRegressor
        using ONLY X_arr[train_idx]/y_arr[train_idx], then evaluates every
        tree of THAT fold-specific forest ONLY on X_arr[val_idx]/
        y_arr[val_idx] -- rows the fold forest never saw during its own
        .fit(). This is called BEFORE the final full-data forest is fit
        (see .fit() below) -- it does not touch self.forest_ at all.

        Returns temporal_mae_i = mean over folds of tree-slot i's
        validation MAE, and also stores the full (n_estimators,
        n_folds) error matrix in self.temporal_fold_tree_errors_ for
        diagnostics.
        """
        n_samples = X_arr.shape[0]

        if n_samples <= self.n_temporal_splits:
            raise ValueError(
                f"weighting_method='temporal' requires more training samples "
                f"({n_samples}) than n_temporal_splits ({self.n_temporal_splits}). "
                "Reduce n_temporal_splits or supply more training data."
            )

        tscv = TimeSeriesSplit(n_splits=self.n_temporal_splits)
        splits = list(tscv.split(X_arr))

        if len(splits) == 0:
            raise ValueError(
                f"TimeSeriesSplit produced zero splits for {n_samples} training "
                f"samples and n_temporal_splits={self.n_temporal_splits}."
            )

        for train_idx, val_idx in splits:
            if len(train_idx) == 0 or len(val_idx) == 0:
                raise ValueError(
                    "A temporal validation fold has zero training or validation "
                    "samples -- increase training data or decrease n_temporal_splits."
                )
            # Explicit chronological safeguard: every fold's training indices
            # must all occur strictly before its validation indices.
            assert train_idx.max() < val_idx.min(), (
                "Chronological split check FAILED for a temporal validation fold: "
                f"train_idx.max()={train_idx.max()} >= val_idx.min()={val_idx.min()}."
            )

        n_trees = self.n_estimators
        fold_tree_errors = np.full((n_trees, len(splits)), np.nan)
        fold_scores = np.full(len(splits), np.nan)

        for fold_i, (train_idx, val_idx) in enumerate(splits):
            X_fold_train = X_arr[train_idx]
            y_fold_train = y_arr[train_idx]
            X_fold_val = X_arr[val_idx]
            y_fold_val = y_arr[val_idx]

            # A COMPLETELY SEPARATE forest, trained ONLY on this fold's
            # training rows -- never on X_fold_val/y_fold_val, and never
            # reused as (or merged into) the final forest.
            fold_forest = self._make_forest()
            fold_forest.fit(X_fold_train, y_fold_train)

            # Prevent any silent n_estimators mismatch between this fold
            # forest and the configured value used everywhere else.
            if len(fold_forest.estimators_) != self.n_estimators:
                raise ValueError(
                    f"Fold {fold_i} forest has {len(fold_forest.estimators_)} trees, "
                    f"expected n_estimators={self.n_estimators}."
                )

            fold_val_errors = np.empty(n_trees)
            for tree_i, tree in enumerate(fold_forest.estimators_):
                pred = tree.predict(X_fold_val)  # ONLY the fold's validation rows
                if not np.all(np.isfinite(pred)):
                    raise ValueError(
                        f"Tree {tree_i} in fold {fold_i}'s fold-specific forest "
                        "produced non-finite predictions on that fold's validation "
                        "rows -- failing clearly rather than silently using a "
                        "fabricated value."
                    )
                mae = mean_absolute_error(y_fold_val, pred)
                fold_tree_errors[tree_i, fold_i] = mae
                fold_val_errors[tree_i] = mae

            fold_scores[fold_i] = fold_val_errors.mean()

        self.temporal_fold_tree_errors_ = fold_tree_errors
        self.temporal_fold_scores_ = fold_scores

        temporal_errors = np.nanmean(fold_tree_errors, axis=1)

        if np.isnan(temporal_errors).any():
            fallback = np.nanmean(temporal_errors)
            if np.isnan(fallback):
                raise ValueError(
                    "All per-tree-slot temporal validation errors are NaN -- "
                    "cannot compute temporal weights."
                )
            temporal_errors = np.where(np.isnan(temporal_errors), fallback, temporal_errors)

        if not np.all(np.isfinite(temporal_errors)):
            raise ValueError("Non-finite temporal MAE encountered after aggregation.")

        return temporal_errors

    def _weights_from_errors(self, errors):
        """Shared error -> normalized weight conversion, used identically
        by both weighting_method values (only the source of `errors`
        differs: OOB MAE vs. cross-fold temporal MAE)."""
        epsilon = 1e-6
        inverse_error = 1.0 / (errors + epsilon)
        raw_weights = inverse_error ** self.weight_power

        weight_sum = raw_weights.sum()
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError(
                f"Computed tree weights are not finite/positive (sum={weight_sum}) "
                "-- cannot normalize."
            )

        weights = raw_weights / weight_sum

        if not np.all(np.isfinite(weights)):
            raise ValueError("Non-finite tree weight(s) after normalization.")
        if np.any(weights < 0):
            raise ValueError("Negative tree weight(s) after normalization.")
        if not np.isclose(weights.sum(), 1.0, atol=1e-6):
            raise ValueError(f"Tree weights do not sum to 1 after normalization (sum={weights.sum()}).")

        return weights

    def fit(self, X_train, y_train):
        """
        Trains the model using whichever weighting_method was configured.
        Receives ONLY X_train/y_train -- never X_test/y_test, which this
        class has no parameter or code path to accept. The final test set
        must only ever be passed to .predict(), outside this module.

        For weighting_method="temporal": genuine out-of-sample temporal
        reliability is estimated FIRST (via independently trained
        fold-specific forests, touching only X_train internally), and
        the final full-data forest is fit AFTER that estimation completes
        (Step 5 in the class design) -- the final forest plays no part in
        computing the temporal error estimates.

        For weighting_method="oob": the final forest is fit first (OOB
        errors are computed FROM its own bootstrap holdout, so it must
        already exist).
        """
        X_arr = np.asarray(X_train)
        y_arr = np.asarray(y_train)

        if self.weighting_method == "temporal":
            # STEPS 1-4: genuine out-of-sample temporal reliability,
            # entirely independent of the final forest.
            errors = self._compute_temporal_errors(X_arr, y_arr)
            self.tree_temporal_errors_ = errors
            self.tree_oob_errors_ = None

            # STEP 5: NOW fit the one final forest on the complete data.
            self.forest_ = self._make_forest()
            self.forest_.fit(X_arr, y_arr)

        elif self.weighting_method == "oob":
            self.forest_ = self._make_forest()
            self.forest_.fit(X_arr, y_arr)
            errors = self._compute_oob_errors(X_arr, y_arr)
            self.tree_oob_errors_ = errors
            self.tree_temporal_errors_ = None

        else:
            raise ValueError(
                f"Unknown weighting_method: {self.weighting_method!r}. "
                "Must be 'oob' or 'temporal'."
            )

        if len(self.forest_.estimators_) != self.n_estimators:
            raise ValueError(
                f"Final forest has {len(self.forest_.estimators_)} trees, "
                f"expected n_estimators={self.n_estimators}."
            )
        if len(errors) != self.n_estimators:
            raise ValueError(
                f"Computed {len(errors)} tree-slot error(s), expected "
                f"n_estimators={self.n_estimators} (tree-slot transfer mismatch)."
            )

        # STEP 7 (temporal) / equivalent step (oob): shared error -> weight formula.
        self.tree_weights_ = self._weights_from_errors(errors)

        return self

    def predict(self, X):
        if self.forest_ is None:
            raise RuntimeError("Call .fit() before .predict().")

        X_arr = np.asarray(X)
        all_tree_preds = np.array([
            tree.predict(X_arr) for tree in self.forest_.estimators_
        ])
        return np.average(all_tree_preds, axis=0, weights=self.tree_weights_)

    @property
    def feature_importances_(self):
        if self.forest_ is None:
            raise RuntimeError("Call .fit() before requesting feature importances.")
        return self.forest_.feature_importances_

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "random_state": self.random_state,
            "weight_power": self.weight_power,
            "weighting_method": self.weighting_method,
            "n_temporal_splits": self.n_temporal_splits,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
