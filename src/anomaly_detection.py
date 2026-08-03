"""
Statistical & Machine Learning Anomaly Detection for Climate Trend Analyzer.

Implements univariate Z-Score thresholding, IQR-based detection,
Isolation Forest ML anomaly detection, and unified anomaly flagging.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import (
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_RANDOM_STATE,
    ZSCORE_THRESHOLD,
)
from src.constants import (
    COL_ANOMALY_COMBINED,
    COL_ANOMALY_IFOREST,
    COL_ANOMALY_ZSCORE,
    COL_HUMIDITY,
    COL_PRECIPITATION,
    COL_SOLAR_RADIATION,
    COL_TEMP_MEAN,
    COL_WIND_SPEED,
)
from src.logger import get_logger, log_pipeline_stage

logger = get_logger("anomaly_detection")

ANOMALY_VARIABLES = [
    COL_TEMP_MEAN,
    COL_PRECIPITATION,
    COL_HUMIDITY,
    COL_SOLAR_RADIATION,
    COL_WIND_SPEED,
]


def detect_zscore_anomalies(
    series: pd.Series,
    threshold: float = ZSCORE_THRESHOLD,
) -> pd.Series:
    """
    Detect anomalies using Z-Score thresholding.

    Flags data points where |Z| > threshold.

    Args:
        series: Numeric time series.
        threshold: Z-Score threshold (default 2.5).

    Returns:
        Boolean Series (True = anomaly).
    """
    mean = series.mean()
    std = series.std()

    if std == 0:
        return pd.Series(False, index=series.index)

    z_scores = np.abs((series - mean) / std)
    anomalies = z_scores > threshold

    return anomalies


def detect_iqr_anomalies(
    series: pd.Series,
    multiplier: float = 1.5,
) -> pd.Series:
    """
    Detect anomalies using the IQR (Interquartile Range) method.

    Args:
        series: Numeric time series.
        multiplier: IQR multiplier (default 1.5).

    Returns:
        Boolean Series (True = anomaly).
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    anomalies = (series < lower_bound) | (series > upper_bound)
    return anomalies


def detect_isolation_forest_anomalies(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    contamination: float = ISOLATION_FOREST_CONTAMINATION,
    random_state: int = ISOLATION_FOREST_RANDOM_STATE,
) -> pd.Series:
    """
    Detect anomalies using Isolation Forest (unsupervised ML).

    Args:
        df: DataFrame with climate variables.
        columns: Columns to use for detection.
        contamination: Expected proportion of anomalies.
        random_state: Random seed for reproducibility.

    Returns:
        Boolean Series (True = anomaly).
    """
    if columns is None:
        columns = [c for c in ANOMALY_VARIABLES if c in df.columns]

    if len(columns) < 2:
        logger.warning("Insufficient columns for Isolation Forest")
        return pd.Series(False, index=df.index)

    # Prepare feature matrix
    feature_df = df[columns].dropna()

    if len(feature_df) < 10:
        logger.warning("Insufficient data for Isolation Forest")
        return pd.Series(False, index=df.index)

    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_df)

    # Fit Isolation Forest
    clf = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )

    predictions = clf.fit_predict(features_scaled)

    # Convert to boolean (IsolationForest: -1 = anomaly, 1 = normal)
    anomaly_mask = predictions == -1

    # Map back to original index
    result = pd.Series(False, index=df.index)
    result.loc[feature_df.index] = anomaly_mask

    return result


def detect_statistical_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalies using Z-Score and IQR methods across all variables.

    Args:
        df: Processed DataFrame.

    Returns:
        DataFrame with anomaly flag columns added.
    """
    df = df.copy()

    # Initialize combined anomaly column
    df[COL_ANOMALY_ZSCORE] = False

    for col in ANOMALY_VARIABLES:
        if col in df.columns:
            # Z-Score anomalies
            zscore_anomalies = detect_zscore_anomalies(df[col])
            df[COL_ANOMALY_ZSCORE] = df[COL_ANOMALY_ZSCORE] | zscore_anomalies

            anomaly_count = zscore_anomalies.sum()
            if anomaly_count > 0:
                logger.info(f"Z-Score anomalies in {col}: {anomaly_count} days")

    return df


def detect_ml_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalies using Isolation Forest.

    Args:
        df: Processed DataFrame.

    Returns:
        DataFrame with ML anomaly flag column added.
    """
    df = df.copy()

    # Isolation Forest anomalies
    iforest_anomalies = detect_isolation_forest_anomalies(df)
    df[COL_ANOMALY_IFOREST] = iforest_anomalies

    anomaly_count = iforest_anomalies.sum()
    logger.info(f"Isolation Forest anomalies: {anomaly_count} days")

    return df


def combine_anomaly_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine statistical and ML anomaly flags into a unified indicator.

    An anomaly is flagged if detected by either method.

    Args:
        df: DataFrame with anomaly flag columns.

    Returns:
        DataFrame with unified anomaly column.
    """
    df = df.copy()

    df[COL_ANOMALY_COMBINED] = False

    if COL_ANOMALY_ZSCORE in df.columns:
        df[COL_ANOMALY_COMBINED] = df[COL_ANOMALY_COMBINED] | df[COL_ANOMALY_ZSCORE]

    if COL_ANOMALY_IFOREST in df.columns:
        df[COL_ANOMALY_COMBINED] = df[COL_ANOMALY_COMBINED] | df[COL_ANOMALY_IFOREST]

    combined_count = df[COL_ANOMALY_COMBINED].sum()
    logger.info(f"Combined anomalies: {combined_count} days "
                f"({combined_count / len(df) * 100:.2f}%)")

    return df


def compute_anomaly_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute summary statistics for detected anomalies.

    Returns:
        Dictionary with anomaly summary metrics.
    """
    summary = {}

    if COL_ANOMALY_COMBINED in df.columns:
        anomaly_df = df[df[COL_ANOMALY_COMBINED]]

        summary["total_anomaly_days"] = int(len(anomaly_df))
        summary["anomaly_percentage"] = round(
            len(anomaly_df) / len(df) * 100, 2
        ) if len(df) > 0 else 0

        if COL_TEMP_MEAN in df.columns:
            summary["anomaly_temperature_stats"] = {
                "mean": float(anomaly_df[COL_TEMP_MEAN].mean()) if len(anomaly_df) > 0 else None,
                "max": float(anomaly_df[COL_TEMP_MEAN].max()) if len(anomaly_df) > 0 else None,
                "min": float(anomaly_df[COL_TEMP_MEAN].min()) if len(anomaly_df) > 0 else None,
            }

    if COL_ANOMALY_ZSCORE in df.columns:
        summary["zscore_anomaly_days"] = int(df[COL_ANOMALY_ZSCORE].sum())

    if COL_ANOMALY_IFOREST in df.columns:
        summary["iforest_anomaly_days"] = int(df[COL_ANOMALY_IFOREST].sum())

    return summary


def run_anomaly_detection(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Execute the complete anomaly detection pipeline.

    Steps:
        1. Statistical (Z-Score) anomaly detection
        2. ML (Isolation Forest) anomaly detection
        3. Combine flags into unified indicator

    Args:
        df: Processed DataFrame.

    Returns:
        Tuple of (DataFrame with anomaly flags, anomaly summary).
    """
    log_pipeline_stage("Anomaly Detection")

    df = detect_statistical_anomalies(df)
    df = detect_ml_anomalies(df)
    df = combine_anomaly_flags(df)

    summary = compute_anomaly_summary(df)

    logger.info(f"Anomaly detection complete: {summary.get('total_anomaly_days', 0)} anomaly days")
    return df, summary
