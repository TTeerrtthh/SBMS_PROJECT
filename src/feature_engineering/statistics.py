import numpy as np


def calculate_statistics(series):
    """
    Calculate statistical features from one sensor column.
    """

    return {
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
        "median": series.median(),
        "variance": series.var(),
        "skewness": series.skew(),
        "kurtosis": series.kurt(),
    }


def rms(series):
    """
    Root Mean Square
    """

    return np.sqrt(np.mean(series ** 2))


def peak_to_peak(series):
    """
    Peak-to-Peak Value
    """

    return series.max() - series.min()


def energy(series):
    """
    Signal Energy
    """

    return np.sum(series ** 2)


def extract_numeric_features(series):
    """
    Complete feature extraction for one sensor.
    """

    features = calculate_statistics(series)

    features["rms"] = rms(series)
    features["peak_to_peak"] = peak_to_peak(series)
    features["energy"] = energy(series)

    return features