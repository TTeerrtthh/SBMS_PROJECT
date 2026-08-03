"""
src/preprocessing/cleaner.py

Cleans individual cycle files (missing values, duplicate rows, bad dtypes)
and validates that the already-segregated dataset/processed/ folder
structure is complete before feature engineering begins.

Note: since the raw NASA data has already been manually segregated into
dataset/processed/<battery_id>/{charge,discharge,impedance}/, this module
does NOT do the sorting itself -- it only cleans and validates what's
already there.
"""

import pandas as pd
from src.core.config import PROCESSED_DATA, BATTERY_IDS
from src.core.data_loader import list_processed_files, load_processed_file


def clean_cycle_dataframe(df, numeric_columns=None):
    """
    Cleans a single cycle's raw DataFrame:
      1. Drops fully-empty rows.
      2. Converts specified columns to numeric, coercing bad values to NaN.
      3. Forward/backward-fills small gaps of missing sensor readings
         (a handful of missing samples within an otherwise valid cycle is
         normal sensor noise, not a reason to discard the whole cycle).
      4. Drops any row still containing NaN after filling (only happens if
         an entire column was missing, which is rare and worth noticing).
    """
    df = df.copy()
    df = df.dropna(how="all")

    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_columns] = df[numeric_columns].ffill().bfill()
    df = df.dropna(subset=numeric_columns)

    return df.reset_index(drop=True)


def validate_processed_structure():
    """
    Checks that every battery in BATTERY_IDS has at least one file in each
    of charge/discharge/impedance. Prints a clear report -- run this first
    in Notebook 02 before anything else, so a missing folder is caught
    immediately instead of causing a confusing error three notebooks later.
    """
    report = {}
    all_ok = True

    for battery_id in BATTERY_IDS:
        report[battery_id] = {}
        for experiment in ["charge", "discharge", "impedance"]:
            try:
                files = list_processed_files(battery_id, experiment)
                count = len(files)
            except FileNotFoundError:
                count = 0
            report[battery_id][experiment] = count
            if count == 0:
                all_ok = False

    print("Processed data structure check:")
    for battery_id, experiments in report.items():
        print(f"  {battery_id}:")
        for experiment, count in experiments.items():
            flag = "OK" if count > 0 else "MISSING/EMPTY"
            print(f"    {experiment:<12} -> {count:>4} files   [{flag}]")

    if all_ok:
        print("\nAll expected folders contain data. Safe to proceed.")
    else:
        print("\nWARNING: at least one experiment folder is empty. "
              "Fix this before running feature engineering.")

    return report


def clean_all_files(battery_id, experiment, numeric_columns=None):
    """
    Loads and cleans every file for one battery + experiment type,
    returning a dict {filename: cleaned_df}. Use this in Notebook 02
    to check cleaning works before Notebook 03 uses it inside feature
    extraction.
    """
    filenames = list_processed_files(battery_id, experiment)
    cleaned = {}
    for fname in filenames:
        df = load_processed_file(battery_id, experiment, fname)
        cleaned[fname] = clean_cycle_dataframe(df, numeric_columns=numeric_columns)
    return cleaned
