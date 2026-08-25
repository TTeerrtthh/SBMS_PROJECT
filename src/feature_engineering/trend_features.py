"""
src/feature_engineering/trend_features.py

Adds CROSS-CYCLE (trajectory) features on top of the per-cycle statistics
from extractor.py. This is the piece that was missing: extractor.py only
describes what happens WITHIN one cycle; this module describes how things
CHANGE ACROSS cycles, which is what SOH/RUL prediction most needs.

Three feature groups, each with a clear physical justification:

  1. Capacity fade rate   -- how fast is capacity dropping right now
  2. Rolling mean/variance -- is the battery's behavior becoming more
                               erratic over recent cycles (rolling std),
                               and what's the recent trend level (rolling mean)
  3. Internal resistance trend -- pulled from the impedance test files
                                   (Re, Rct), which reflect a different,
                                   independent physical degradation
                                   mechanism than capacity fade
"""

import re
import numpy as np
import pandas as pd

_DIGITS_RE = re.compile(r"\d+")


def _extract_numeric_id(filename):
    """
    NASA's per-cycle files are numbered with one shared, incrementing
    counter across ALL experiment types (charge/discharge/impedance) for
    a given battery -- e.g. '05121.csv', '05122.csv', '05123.csv' might be
    charge/discharge/impedance in sequence. Extracting this number lets us
    line up an impedance reading with the nearest discharge cycle in time,
    without needing to parse the messy start_time array for this purpose.
    """
    match = _DIGITS_RE.search(filename)
    return int(match.group()) if match else np.nan


def add_capacity_fade_rate(df, capacity_col="Capacity", cycle_col="cycle_index", window=5):
    """
    Adds 'capacity_fade_rate': the slope of capacity vs. cycle index over
    the trailing `window` cycles (a rolling linear regression slope, not
    just a two-point difference, so it's more robust to single-cycle noise
    and the well-known capacity-regeneration bumps in this dataset).

    A more negative value = faster degradation happening right now.
    The first (window-1) cycles won't have enough history for a full
    window -- their fade rate is computed from however many prior cycles
    are available (minimum 2), and the very first cycle gets NaN, which
    we fill with 0 (no fade detected yet, the most reasonable default).
    """
    df = df.sort_values(cycle_col).reset_index(drop=True)
    capacities = df[capacity_col].to_numpy()
    cycles = df[cycle_col].to_numpy()

    fade_rates = np.full(len(df), np.nan)

    for i in range(len(df)):
        start = max(0, i - window + 1)
        if i - start < 1:  # need at least 2 points to fit a slope
            continue
        x = cycles[start:i + 1]
        y = capacities[start:i + 1]
        slope, _ = np.polyfit(x, y, 1)
        fade_rates[i] = slope

    df["capacity_fade_rate"] = pd.Series(fade_rates).fillna(0.0)
    return df


def add_rolling_features(df, columns, cycle_col="cycle_index", window=5):
    """
    Adds rolling mean and rolling std (a rolling-variance style measure of
    recent instability) for each column in `columns`, computed over the
    trailing `window` cycles. Column names get suffixes:
        '<col>_rolling_mean_w<window>'
        '<col>_rolling_std_w<window>'

    Rolling std answers a different question than the per-cycle std already
    computed in extractor.py: extractor.py's std describes variability
    WITHIN one cycle's raw signal; this rolling std describes how much a
    per-cycle SUMMARY statistic (e.g. mean voltage) is fluctuating ACROSS
    recent cycles -- a rise here can indicate the battery entering a more
    unstable degradation phase.
    """
    df = df.sort_values(cycle_col).reset_index(drop=True)

    for col in columns:
        if col not in df.columns:
            continue
        df[f"{col}_rolling_mean_w{window}"] = df[col].rolling(window=window, min_periods=1).mean()
        df[f"{col}_rolling_std_w{window}"] = df[col].rolling(window=window, min_periods=1).std().fillna(0.0)

    return df


def add_impedance_trend_features(df, metadata, battery_id, cycle_col="cycle_index", window=5):
    """
    Pulls internal resistance (Re) and charge transfer resistance (Rct)
    from the impedance test metadata, and attaches the MOST RECENT
    impedance reading available (in true test order, using the shared
    NASA file-numbering scheme) to each discharge cycle -- since impedance
    is measured periodically, not on every single discharge cycle.

    Adds 4 columns: 'Re_nearest', 'Rct_nearest' (the raw attached values),
    plus 'Re_trend' and 'Rct_trend' (rolling slope of each over the
    trailing `window` discharge cycles) -- this trend is the actual
    "internal resistance growth" signal that matters for SOH.

    Discharge cycles that occur BEFORE the first available impedance
    measurement for this battery will have Re_nearest/Rct_nearest/
    Re_trend/Rct_trend left as NaN (genuinely unavailable), rather than
    filled from a later impedance reading -- filling from a later reading
    would let an earlier cycle's features depend on information that did
    not exist yet at that point in the battery's life, i.e. temporal
    (future-data) leakage. Downstream code must handle these NaNs
    explicitly (e.g. exclude affected rows/columns, or impute using only
    past information) rather than assume this function always returns a
    fully populated column.

    If this battery has no impedance data available, all four columns are
    filled with 0 and a warning is printed -- feature engineering still
    completes, just without this signal.
    """
    df = df.sort_values(cycle_col).reset_index(drop=True)

    impedance_meta = metadata[
        (metadata["battery_id"] == battery_id) & (metadata["type"] == "impedance")
    ].copy()

    if len(impedance_meta) == 0 or "Re" not in impedance_meta.columns:
        print(f"Warning: no usable impedance data found for {battery_id}. "
              "Re_nearest/Rct_nearest/Re_trend/Rct_trend will be filled with 0.")
        df["Re_nearest"] = 0.0
        df["Rct_nearest"] = 0.0
        df["Re_trend"] = 0.0
        df["Rct_trend"] = 0.0
        return df

    impedance_meta["numeric_id"] = impedance_meta["filename"].apply(_extract_numeric_id)
    impedance_meta = impedance_meta.dropna(subset=["numeric_id", "Re", "Rct"])
    impedance_meta = impedance_meta.sort_values("numeric_id")

    df["numeric_id"] = df["File"].apply(_extract_numeric_id)
    df = df.sort_values("numeric_id")

    merged = pd.merge_asof(
        df, impedance_meta[["numeric_id", "Re", "Rct"]],
        on="numeric_id", direction="backward",
    )
    merged = merged.rename(columns={"Re": "Re_nearest", "Rct": "Rct_nearest"})

    # Any discharge cycle that occurs BEFORE the first available impedance
    # reading has no valid impedance measurement yet. Do NOT backfill/forward-fill
    # a value from a LATER impedance test into these earlier cycles -- doing so
    # would mean an earlier row's feature depends on information that did not
    # exist yet at that point in time, i.e. temporal (future-data) leakage.
    # merge_asof(direction="backward") already leaves these rows as NaN;
    # we deliberately leave them as NaN here rather than filling them with
    # any value (0.0 or bfilled), so they are honestly represented as
    # "unavailable" rather than fabricated.
    merged = merged.sort_values(cycle_col).reset_index(drop=True)

    for col in ["Re_nearest", "Rct_nearest"]:
        trend_col = col.replace("_nearest", "_trend")
        slopes = np.full(len(merged), np.nan)
        values = merged[col].to_numpy()
        cycles = merged[cycle_col].to_numpy()
        for i in range(len(merged)):
            if np.isnan(values[i]):
                # The current cycle itself has no valid impedance reading yet
                # (before the first impedance test) -- a trend cannot be
                # defined for a value that doesn't exist. Leave as NaN.
                continue
            start = max(0, i - window + 1)
            window_cycles = cycles[start:i + 1]
            window_values = values[start:i + 1]
            # Use only the REAL, already-observed values within the trailing
            # window (dropping any NaN, e.g. the boundary just after the
            # first impedance reading appears) -- never a future value, and
            # never a fabricated one. Require at least 2 valid points to fit
            # a slope, same minimum used by add_capacity_fade_rate() above.
            valid_mask = ~np.isnan(window_values)
            if valid_mask.sum() < 2:
                continue
            slope, _ = np.polyfit(window_cycles[valid_mask], window_values[valid_mask], 1)
            slopes[i] = slope
        # NOTE: no fillna(0.0) here -- a missing trend must remain NaN/
        # unavailable, not be silently reported as "zero change", which
        # would misrepresent an unknown value as a known, neutral one.
        merged[trend_col] = pd.Series(slopes)

    return merged.drop(columns=["numeric_id"])
