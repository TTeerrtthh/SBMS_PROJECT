"""
src/explainability/shap_utils.py

Small helper functions for running SHAP on the trained models.

Usage (from a notebook):

    from src.explainability.shap_utils import get_shap_values, plot_shap_summary

    # NOTE: AdaptiveWeightedRF is a wrapper around a RandomForestRegressor,
    # not a tree model itself -- SHAP's TreeExplainer needs the underlying
    # forest, so pass `ahrf_model.forest_`, not `ahrf_model` directly.
    explainer, shap_values = get_shap_values(ahrf_model.forest_, X_test)
    plot_shap_summary(shap_values, X_test)
"""

import shap
import matplotlib.pyplot as plt


def get_shap_values(model, X, sample_size=None, random_state=42):
    """
    Computes SHAP values for a tree-based model (Random Forest, LightGBM,
    XGBoost, CatBoost all supported via shap.TreeExplainer).

    model : a fitted tree-based model. If you're explaining the HybridRFRegressor,
        pass `model.rf_model_` (its Stage 1) -- SHAP works on a single tree
        ensemble, not the 2-stage blend directly. Explain Stage 1 and mention
        in your report that Stage 2 is a small correction on top.
    X : the feature DataFrame to explain.
    sample_size : if your dataset is small (likely, for a single battery),
        leave this as None and use the full X. Only subsample for very large
        datasets where SHAP would be slow.
    """
    if sample_size is not None and len(X) > sample_size:
        X = X.sample(sample_size, random_state=random_state)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return explainer, shap_values


def plot_shap_summary(shap_values, X, max_display=15, title="SHAP Summary"):
    """Global feature importance + effect direction, across all samples."""
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_shap_bar(shap_values, X, max_display=15, title="SHAP Mean |Impact|"):
    """Simpler bar-chart version -- easier to read in a viva/presentation
    than the beeswarm summary plot."""
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_shap_dependence(feature_name, shap_values, X, interaction_index="auto"):
    """
    Shows how a single feature's effect on the prediction changes across its
    range, optionally colored by the feature it interacts with most.
    Use this for 1-2 of your top features from the summary plot -- not all
    of them, to keep the report focused.
    """
    shap.dependence_plot(
        feature_name, shap_values, X,
        interaction_index=interaction_index,
        show=False,
    )
    plt.tight_layout()
    plt.show()


def explain_single_prediction(explainer, shap_values, X, row_index=0):
    """
    Local explanation -- why did the model make THIS specific prediction for
    one row. Useful for a "case study" figure in your report (e.g. "here's
    why the model predicted low SOH for this particular cycle").
    """
    shap.force_plot(
        explainer.expected_value,
        shap_values[row_index, :],
        X.iloc[row_index, :],
        matplotlib=True,
        show=False,
    )
    plt.tight_layout()
    plt.show()
