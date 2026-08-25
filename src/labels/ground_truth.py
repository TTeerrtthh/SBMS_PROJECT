"""
src/labels/ground_truth.py

Generates the three target labels: SOC, SOH, RUL.
"""

import numpy as np
import pandas as pd
from src.core.config import RATED_CAPACITY_AH, EOL_SOH_THRESHOLD


def generate_soc_labels(discharge_features, voltage_col="Voltage_measured_mean"):
    """
    SOC proxy label: min-max normalized mean voltage per cycle, scaled to 0-100.

    IMPORTANT / HONEST LIMITATION (state this in your report): this is a
    voltage-based PROXY for SOC, not a true Coulomb-counted SOC, since NASA's
    dataset doesn't provide a direct SOC ground truth. This is a common and
    accepted approach in student-level battery-ML projects, but it should be
    described accurately, not implied to be a precise physical measurement.
    """
    df = discharge_features.copy()
    v = df[voltage_col]
    df["SOC"] = (v - v.min()) / (v.max() - v.min()) * 100
    return df


def generate_soh_labels(discharge_features, metadata, rated_capacity_ah=None,
                         merge_left_on="File", merge_right_on="filename",
                         capacity_col="Capacity"):
    """
    SOH label: SOH_n = (measured capacity at cycle n / rated capacity) * 100.

    discharge_features : output of build_feature_dataset() for the discharge
        experiment -- must contain a file-identifier column (default 'File').
        If a 'Capacity' column is ALREADY present (e.g. because Notebook 03's
        trend-feature step already merged it in to compute capacity_fade_rate),
        that existing column is used directly and metadata is NOT re-merged
        -- avoiding duplicate/conflicting columns.
    metadata : output of load_metadata() -- must contain a matching filename
        column (default 'filename') and a Capacity column. Only used if
        discharge_features doesn't already have a Capacity column.
    """
    if rated_capacity_ah is None:
        rated_capacity_ah = RATED_CAPACITY_AH

    if capacity_col in discharge_features.columns:
        merged = discharge_features.copy()
    else:
        meta_subset = metadata[[merge_right_on, capacity_col]].drop_duplicates()
        merged = discharge_features.merge(
            meta_subset,
            left_on=merge_left_on,
            right_on=merge_right_on,
            how="left",
        )

    missing = merged[capacity_col].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} rows had no matching Capacity value "
              f"after merging on '{merge_left_on}' <-> '{merge_right_on}'. "
              "Check that filenames match exactly between your features and metadata.")

    merged["SOH (%)"] = (merged[capacity_col] / rated_capacity_ah) * 100

    return merged


def generate_rul_labels(soh_df, soh_col="SOH (%)", cycle_col="cycle_index",
                         battery_col="battery_id", eol_threshold=None):
    """
    RUL label per cycle: number of cycles remaining until SOH first drops
    to/below the End-of-Life threshold (default 80%), computed SEPARATELY
    for each battery_id (important once you add more than one battery --
    RUL must never be computed across batteries mixed together).

    For cycles that occur AFTER the battery has already crossed the
    threshold, RUL is clipped to 0 (already at/past end-of-life).

    If a battery's SOH never drops to the threshold within the available
    data, RUL is computed relative to the last available cycle and a
    warning is printed -- this is a real limitation (censored data) worth
    mentioning explicitly in your report.
    """
    if eol_threshold is None:
        eol_threshold = EOL_SOH_THRESHOLD

    df = soh_df.copy()
    df["RUL"] = np.nan

    for battery_id, group in df.groupby(battery_col):
        group = group.sort_values(cycle_col)
        below_threshold = group[group[soh_col] <= eol_threshold]

        if len(below_threshold) == 0:
            print(f"Note: battery {battery_id} never reached the "
                  f"{eol_threshold}% SOH threshold in this dataset. "
                  "RUL for this battery is computed relative to its last "
                  "recorded cycle -- treat these RUL values as a lower "
                  "bound, not a confirmed end-of-life point.")
            eol_cycle = group[cycle_col].max()
        else:
            eol_cycle = below_threshold[cycle_col].min()

        rul_values = (eol_cycle - group[cycle_col]).clip(lower=0)
        df.loc[group.index, "RUL"] = rul_values.values

    return df
