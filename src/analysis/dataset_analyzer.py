from pathlib import Path
import pandas as pd

from src.core.config import SELECTED_DATA


def analyze_battery(battery="B0005"):
    """
    Analyze all CSV files inside a battery folder.
    """

    battery_path = SELECTED_DATA / battery

    csv_files = sorted(battery_path.glob("*.csv"))

    print("=" * 60)
    print(f"Battery : {battery}")
    print(f"Total CSV Files : {len(csv_files)}")
    print("=" * 60)

    return csv_files

def count_rows(csv_files):

    rows = []

    for file in csv_files:

        df = pd.read_csv(file)

        rows.append(len(df))

    return rows
