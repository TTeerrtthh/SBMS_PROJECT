"""
src/feature_engineering/extractor.py

Turns raw per-cycle time-series data (voltage, current, temperature over
time within one discharge/charge cycle) into a fixed-size row of
statistical features. This is what makes each cycle usable as one row
of input to a Random Forest, instead of a variable-length sequence.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from src.core.data_loader import list_processed_files, load_processed_file

# Columns we compute statistics for. Adjust this list if your actual
# NASA CSV columns use different names -- check with df.columns in
# Notebook 01 first and update here if needed (this is the ONE place
# to change it).
DEFAULT_SIGNAL_COLUMNS = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Voltage_load",
    "Current_load",
    "Time",
]


def compute_statistics(series):
    """
    Computes 11 summary statistics for a single numeric column (one signal,
    one cycle): mean, std, variance, median, rms, min, max, skewness,
    kurtosis, peak-to-peak, and energy.

    Why these specifically: mean/median/min/max/std/variance describe the
    signal's central tendency and spread; RMS and energy relate to the
    physical power/energy delivered during the cycle; skewness and kurtosis
    capture the shape of the distribution (e.g. a skewed voltage curve can
    indicate an unusual discharge pattern); peak-to-peak captures the total
    swing, useful for spotting instability.
    """
    values = series.dropna().to_numpy(dtype=float)

    if len(values) == 0:
        # Return NaNs rather than crashing -- lets us spot and handle
        # empty/corrupt cycles explicitly in the notebook instead of
        # silently losing them.
        stats = {name: np.nan for name in
                 ["mean", "std", "var", "median", "rms", "min", "max",
                  "skew", "kurtosis", "peak_to_peak", "energy"]}
        return stats

    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "var": np.var(values),
        "median": np.median(values),
        "rms": np.sqrt(np.mean(values ** 2)),
        "min": np.min(values),
        "max": np.max(values),
        "skew": skew(values) if len(values) > 2 else np.nan,
        "kurtosis": kurtosis(values) if len(values) > 2 else np.nan,
        "peak_to_peak": np.max(values) - np.min(values),
        "energy": np.sum(values ** 2),
    }


def extract_features_from_cycle(df, signal_columns=None):
    """
    Given one cycle's raw DataFrame, returns a flat dict of features:
    {'Voltage_measured_mean': ..., 'Voltage_measured_std': ..., ...}
    for every column in signal_columns that is actually present in df.
    """
    if signal_columns is None:
        signal_columns = DEFAULT_SIGNAL_COLUMNS

    features = {}
    for col in signal_columns:
        if col not in df.columns:
            continue  # silently skip columns that don't exist in this file
        stats = compute_statistics(df[col])
        for stat_name, value in stats.items():
            features[f"{col}_{stat_name}"] = value

    return features


def build_feature_dataset(battery_id, experiment, signal_columns=None):
    """
    Loops through every processed cycle file for one battery + experiment
    type, extracts features from each, and returns one combined DataFrame
    -- one row per cycle file, with a 'File' and 'Experiment' identifier
    column preserved so it can be merged with metadata/labels later.
    """
    filenames = list_processed_files(battery_id, experiment)

    rows = []
    for fname in filenames:
        df = load_processed_file(battery_id, experiment, fname)
        features = extract_features_from_cycle(df, signal_columns=signal_columns)
        features["File"] = fname
        features["Experiment"] = experiment
        features["battery_id"] = battery_id
        rows.append(features)

    result = pd.DataFrame(rows)

    # Preserve file order as a cycle-order proxy. IMPORTANT: verify in
    # Notebook 01 that filenames sort into true chronological order for
    # your specific dataset (e.g. zero-padded numeric names like
    # '00001.csv', '00002.csv' sort correctly; non-padded names like
    # '1.csv', '10.csv', '2.csv' do NOT sort correctly as strings).
    result = result.sort_values("File").reset_index(drop=True)
    result["cycle_index"] = np.arange(1, len(result) + 1)

    return result
