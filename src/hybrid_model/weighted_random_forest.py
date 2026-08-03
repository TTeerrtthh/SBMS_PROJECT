"""
src/hybrid_model/weighted_random_forest.py

The core AHRF contribution: a Random Forest where each tree's vote is
weighted by that tree's OUT-OF-BAG (OOB) accuracy, instead of every tree
getting an equal vote (which is what plain RandomForestRegressor does).

Why OOB instead of a separate validation split: Random Forest already
leaves ~1/3 of the training data unseen by each individual tree, due to
bootstrap sampling. We use that "free" held-out data per tree to score it,
instead of manually reserving a validation set -- which would shrink the
training data available to grow the forest in the first place. This makes
AHRF directly comparable to a plain RF trained on the exact same data,
with no unfair data disadvantage.

IMPLEMENTATION NOTE: earlier versions of this file used sklearn's private
`sklearn.ensemble._forest._generate_sample_indices()` to reconstruct each
tree's bootstrap sample. That function's signature changed between sklearn
versions (some versions require an extra `sample_weight` argument), which
caused a TypeError on some setups. This version reimplements the same
bootstrap logic directly using `sklearn.utils.check_random_state`, a
stable, public utility -- so this class no longer depends on any sklearn
internal/private API and will not break across sklearn versions.

Usage (from a notebook):

    from src.hybrid_model.weighted_random_forest import AdaptiveWeightedRF

    model = AdaptiveWeightedRF(**best_params_from_optuna)
    model.fit(X_train, y_train)          # single call, no separate val set needed
    y_pred = model.predict(X_test)

    print(model.tree_weights_)  # inspect which trees mattered most
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
    trees' predictions, where each tree's weight is based on its own
    out-of-bag (OOB) accuracy -- the tree's error on the training rows
    IT never saw during its own bootstrap sampling.

    Plain RandomForestRegressor prediction:
        y_pred = mean(tree_1(x), tree_2(x), ..., tree_N(x))

    AdaptiveWeightedRF prediction:
        y_pred = sum(w_i * tree_i(x))  where sum(w_i) = 1
        and w_i is higher for trees with lower OOB error.
    """

    def __init__(self, n_estimators=200, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", random_state=42,
                 weight_power=1.0):
        """
        weight_power : controls how aggressively good trees are favored.
            0.0 -> all trees weighted equally (mathematically reduces to plain RF).
            1.0 -> weight directly proportional to inverse OOB error (default).
            Higher values (2.0-4.0) favor the best trees more strongly.
            Tune this alongside the usual RF parameters via Optuna.

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

        self.forest_ = None
        self.tree_weights_ = None
        self.tree_oob_errors_ = None

    def fit(self, X_train, y_train):
        """
        Trains the forest AND computes per-tree OOB-based weights in one
        call -- no separate validation set required.
        """
        X_arr = np.asarray(X_train)
        y_arr = np.asarray(y_train)
        n_samples = X_arr.shape[0]

        self.forest_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=True,   # required for OOB weighting
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.forest_.fit(X_arr, y_arr)

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

        epsilon = 1e-6
        inverse_error = 1.0 / (errors + epsilon)
        raw_weights = inverse_error ** self.weight_power
        self.tree_weights_ = raw_weights / raw_weights.sum()
        self.tree_oob_errors_ = errors

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
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
