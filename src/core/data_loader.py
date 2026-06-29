from pathlib import Path
import pandas as pd

from src.core.config import SELECTED_DATA


def load_csv(filename: str, battery: str = "B0005"):
    """
    Load a CSV file from the selected battery dataset.

    Parameters:
        filename (str): CSV filename (e.g., '05122.csv')
        battery (str): Battery folder name (default: B0005)

    Returns:
        pandas.DataFrame
    """

    file_path = SELECTED_DATA / battery / filename

    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found.")

    return pd.read_csv(file_path)