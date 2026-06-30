import pandas as pd
from pathlib import Path

from src.core.config import (
    SELECTED_DATA,
    PROCESSED_DATA,
)

# ============================================================
# DATA LOADING
# ============================================================

def load_dataframe(file_path: Path):
    """
    Load CSV file.
    """
    return pd.read_csv(file_path)


# ============================================================
# DATA CLEANING
# ============================================================

def check_missing_values(df):
    """
    Returns number of missing values.
    """
    return df.isnull().sum()


def remove_duplicates(df):
    """
    Remove duplicate rows.
    """
    return df.drop_duplicates()


# ============================================================
# EXPERIMENT TYPE DETECTION
# ============================================================

def detect_experiment_type(df):
    """
    Detect whether CSV belongs to

    - Charge
    - Discharge
    - Impedance
    """

    columns = set(df.columns)

    # ---------------------------
    # DISCHARGE
    # ---------------------------
    if (
        "Voltage_load" in columns
        and "Current_load" in columns
    ):
        return "discharge"

    # ---------------------------
    # CHARGE
    # ---------------------------
    elif (
        "Voltage_charge" in columns
        and "Current_charge" in columns
    ):
        return "charge"

    # ---------------------------
    # IMPEDANCE
    # ---------------------------
    elif (
        "Battery_impedance" in columns
        or "Rectified_Impedance" in columns
    ):
        return "impedance"

    return "unknown"


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def clean_discharge(df):
    """
    Cleaning rules for discharge files.
    """

    df = remove_duplicates(df)

    return df


def clean_charge(df):
    """
    Cleaning rules for charge files.
    """

    df = remove_duplicates(df)

    return df


def clean_impedance(df):
    """
    Cleaning rules for impedance files.
    """

    df = remove_duplicates(df)

    return df


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def process_battery(battery_name="B0005"):
    """
    Process every CSV file of one battery.
    """

    input_folder = SELECTED_DATA / battery_name

    battery_output = PROCESSED_DATA / battery_name

    discharge_folder = battery_output / "discharge"
    charge_folder = battery_output / "charge"
    impedance_folder = battery_output / "impedance"

    discharge_folder.mkdir(parents=True, exist_ok=True)
    charge_folder.mkdir(parents=True, exist_ok=True)
    impedance_folder.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_folder.glob("*.csv"))

    print("=" * 60)
    print(f"Battery : {battery_name}")
    print(f"Total CSV Files : {len(csv_files)}")
    print("=" * 60)

    discharge_count = 0
    charge_count = 0
    impedance_count = 0
    unknown_count = 0

    for file in csv_files:

        print(f"Processing {file.name}")

        df = load_dataframe(file)

        experiment = detect_experiment_type(df)

        # -------------------------
        # DISCHARGE
        # -------------------------
        if experiment == "discharge":

            df = clean_discharge(df)

            output_path = discharge_folder / file.name

            discharge_count += 1

        # -------------------------
        # CHARGE
        # -------------------------
        elif experiment == "charge":

            df = clean_charge(df)

            output_path = charge_folder / file.name

            charge_count += 1

        # -------------------------
        # IMPEDANCE
        # -------------------------
        elif experiment == "impedance":

            df = clean_impedance(df)

            output_path = impedance_folder / file.name

            impedance_count += 1

        # -------------------------
        # UNKNOWN
        # -------------------------
        else:

            print(f"Unknown Experiment : {file.name}")

            unknown_count += 1

            continue

        df.to_csv(output_path, index=False)

    print("\n")
    print("=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)

    print(f"Discharge Files : {discharge_count}")
    print(f"Charge Files    : {charge_count}")
    print(f"Impedance Files : {impedance_count}")
    print(f"Unknown Files   : {unknown_count}")

    print("=" * 60)