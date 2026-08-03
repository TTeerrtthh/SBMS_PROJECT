"""
src/decision_engine/rules.py

A simple, transparent rule-based decision layer that turns SOC/SOH/RUL/
anomaly predictions into a human-readable battery status and maintenance
recommendation. Deliberately kept as plain if/else logic, not another ML
model -- the point of this stage is interpretability, not accuracy.
"""

import pandas as pd


def classify_battery_status(soc, soh, rul, anomaly_flag=False,
                             soh_critical=70, soh_warning=80,
                             rul_critical=10, rul_warning=30):
    """
    Given one battery's current predicted SOC, SOH, RUL (and optionally an
    anomaly flag from residual-based anomaly detection), returns a status
    label and a maintenance recommendation string.

    Thresholds are simple and explainable -- exactly what a guide/examiner
    wants to see at this stage, versus a black-box classifier.
    """
    if anomaly_flag:
        return {
            "status": "ANOMALY DETECTED",
            "recommendation": "Immediate inspection recommended -- prediction "
                               "residual is outside the expected range, "
                               "suggesting unusual battery behavior.",
        }

    if soh <= soh_critical or rul <= rul_critical:
        return {
            "status": "CRITICAL",
            "recommendation": f"Battery health is low (SOH={soh:.1f}%) and/or "
                               f"remaining life is short (RUL={rul:.0f} cycles). "
                               "Schedule replacement soon.",
        }

    if soh <= soh_warning or rul <= rul_warning:
        return {
            "status": "WARNING",
            "recommendation": f"Battery is degrading (SOH={soh:.1f}%, "
                               f"RUL={rul:.0f} cycles). Plan maintenance "
                               "within the next scheduling window.",
        }

    return {
        "status": "HEALTHY",
        "recommendation": f"Battery is within normal operating range "
                           f"(SOH={soh:.1f}%, RUL={rul:.0f} cycles, "
                           f"SOC={soc:.1f}%). No action needed.",
    }


def run_decision_engine(predictions_df, soc_col="SOC_pred", soh_col="SOH_pred",
                         rul_col="RUL_pred", anomaly_col=None):
    """
    Applies classify_battery_status() row-by-row to a DataFrame of
    predictions, returning the DataFrame with 'status' and 'recommendation'
    columns added.

    predictions_df must contain at least soc_col, soh_col, rul_col.
    anomaly_col is optional -- pass its name if you built the anomaly
    detection stage; otherwise every row is treated as non-anomalous.
    """
    statuses = []
    recommendations = []

    for _, row in predictions_df.iterrows():
        anomaly_flag = bool(row[anomaly_col]) if anomaly_col else False
        result = classify_battery_status(
            soc=row[soc_col],
            soh=row[soh_col],
            rul=row[rul_col],
            anomaly_flag=anomaly_flag,
        )
        statuses.append(result["status"])
        recommendations.append(result["recommendation"])

    output = predictions_df.copy()
    output["status"] = statuses
    output["recommendation"] = recommendations
    return output
