from pathlib import Path
import pandas as pd

from src.feature_engineering.statistics import extract_numeric_features
from src.core.config import PROCESSED_DATA, FEATURE_DATA


def extract_features(df):
    """
    Extract statistical features from one dataframe.
    """

    feature_vector = {}

    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:

        features = extract_numeric_features(df[column])

        for feature_name, value in features.items():

            feature_vector[f"{column}_{feature_name}"] = value

    return feature_vector


def process_experiment(experiment_folder: Path, experiment_name: str):
    """
    Process all CSV files belonging to one experiment.
    """

    feature_list = []

    csv_files = sorted(experiment_folder.glob("*.csv"))

    print(f"\nProcessing {experiment_name.upper()}")

    print(f"Total Files : {len(csv_files)}")

    for file in csv_files:

        df = pd.read_csv(file)

        features = extract_features(df)

        features["Experiment"] = experiment_name

        features["File"] = file.name

        feature_list.append(features)

    feature_df = pd.DataFrame(feature_list)

    output_file = FEATURE_DATA / f"{experiment_name}_features.csv"

    feature_df.to_csv(output_file, index=False)

    print(f"Saved -> {output_file.name}")

    return feature_df


def build_feature_datasets(battery="B0005"):
    """
    Build separate feature datasets for Charge,
    Discharge and Impedance experiments.
    """

    battery_folder = PROCESSED_DATA / battery

    charge_df = process_experiment(
        battery_folder / "charge",
        "charge"
    )

    discharge_df = process_experiment(
        battery_folder / "discharge",
        "discharge"
    )

    impedance_df = process_experiment(
        battery_folder / "impedance",
        "impedance"
    )

    return charge_df, discharge_df, impedance_df