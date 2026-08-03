"""
src/core/data_loader.py

Loads raw NASA per-cycle CSV files and the metadata.csv index.
Assumes you have ALREADY segregated the raw NASA CSVs into
dataset/raw/nasa_original/ (all individual per-cycle files) and
dataset/raw/metadata/metadata.csv (the index describing each file:
battery_id, filename, type [charge/discharge/impedance], Capacity, etc.)

This module does NOT sort files into charge/discharge/impedance folders --
that happens in src/preprocessing/cleaner.py, using the metadata's "type"
column as the source of truth.
"""

import re
import pandas as pd
from src.core.config import NASA_ORIGINAL, METADATA_DIR, PROCESSED_DATA

# Handles scientific notation (e.g. "1.7921e+01"), which a naive \d+
# regex breaks on -- important since start_time strings contain values
# formatted this way.
_FLOAT_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

_NUMERIC_METADATA_COLUMNS = ["ambient_temperature", "Capacity", "Re", "Rct"]


def load_metadata():
    """
    Loads dataset/raw/metadata/metadata.csv and cleans two known issues
    found in this dataset:
      1. An embedded header row (a literal 'type,start_time,...' row
         sitting inside the data) -- dropped.
      2. The entire file being fully duplicated (every row appears twice)
         -- removed with drop_duplicates().
    Also converts ambient_temperature/Capacity/Re/Rct to proper numeric
    dtypes (they load as text otherwise, because of issue #1 above).

    Expected columns: type, start_time, ambient_temperature, battery_id,
    test_id, uid, filename, Capacity, Re, Rct
    """
    path = METADATA_DIR / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"metadata.csv not found at {path}. "
            "Place your NASA metadata.csv there before running this."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    before = len(df)
    df = df[df["type"] != "type"].copy()          # drop embedded header row
    df = df.drop_duplicates().reset_index(drop=True)  # drop full-file duplication
    after = len(df)

    if before != after:
        print(f"load_metadata(): cleaned {before - after} bad/duplicate rows "
              f"({before} -> {after}).")

    for col in _NUMERIC_METADATA_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def parse_start_time(start_time_str):
    """
    NASA's start_time column stores a stringified array like
    '[2.0080e+03 4.0000e+00 2.0000e+00 1.5000e+01 2.5000e+01 4.1593e+01]'
    representing [year, month, day, hour, minute, second].

    Returns a tuple of 6 floats, suitable for sorting to get TRUE
    chronological order. Verified to match filename-based order 100% of
    the time for B0005 -- use this as a double-check if you add another
    battery later.
    """
    if not isinstance(start_time_str, str):
        return None
    nums = _FLOAT_RE.findall(start_time_str)
    if len(nums) != 6:
        return None
    return tuple(float(n) for n in nums)


def load_cycle_file(filename):
    """
    Loads a single raw per-cycle CSV by filename (e.g. '00001.csv'),
    from dataset/raw/nasa_original/.
    """
    path = NASA_ORIGINAL / filename
    if not path.exists():
        raise FileNotFoundError(f"Cycle file not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def list_available_files():
    """Returns a sorted list of every raw CSV filename present in nasa_original/."""
    return sorted(p.name for p in NASA_ORIGINAL.glob("*.csv"))


# ---------------------------------------------------------------------------
# Functions below assume the data is ALREADY segregated into
# dataset/processed/<battery_id>/{charge,discharge,impedance}/*.csv
# -- which is the case for this project (segregation already done manually).
# ---------------------------------------------------------------------------

def list_processed_files(battery_id, experiment):
    """
    Lists every CSV filename inside dataset/processed/<battery_id>/<experiment>/.
    experiment must be one of: 'charge', 'discharge', 'impedance'.
    """
    folder = PROCESSED_DATA / battery_id / experiment
    if not folder.exists():
        raise FileNotFoundError(f"No such processed folder: {folder}")
    return sorted(p.name for p in folder.glob("*.csv"))


def load_processed_file(battery_id, experiment, filename):
    """Loads one already-segregated cycle CSV."""
    path = PROCESSED_DATA / battery_id / experiment / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_all_processed(battery_id, experiment):
    """
    Loads every cycle file for one battery + experiment type, returning a
    list of (filename, DataFrame) tuples, sorted by filename (which should
    correspond to cycle order -- verify this assumption in Notebook 01 by
    checking whether filenames are zero-padded / sequential).
    """
    filenames = list_processed_files(battery_id, experiment)
    return [(fname, load_processed_file(battery_id, experiment, fname)) for fname in filenames]
