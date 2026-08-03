"""Tests for anomaly detection module."""

import pytest
import pandas as pd
import numpy as np
from src.anomaly_detection import (
    detect_zscore_anomalies,
    detect_iqr_anomalies,
    detect_isolation_forest_anomalies,
    combine_anomaly_flags,
)


@pytest.fixture
def anomaly_df():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    temps = np.random.normal(25, 3, len(dates))
    temps[50] = 50  # Inject extreme anomaly
    temps[100] = -10  # Inject extreme anomaly
    return pd.DataFrame({
        "date": dates,
        "temperature": temps,
        "precipitation": np.random.uniform(0, 10, len(dates)),
        "humidity": np.random.uniform(30, 90, len(dates)),
        "solar_radiation": np.random.uniform(10, 25, len(dates)),
        "wind_speed": np.random.uniform(0, 15, len(dates)),
    })


def test_detect_zscore_anomalies(anomaly_df):
    anomalies = detect_zscore_anomalies(anomaly_df["temperature"], threshold=2.5)
    assert isinstance(anomalies, pd.Series)
    assert anomalies.sum() >= 2  # Should detect injected anomalies


def test_detect_iqr_anomalies(anomaly_df):
    anomalies = detect_iqr_anomalies(anomaly_df["temperature"])
    assert isinstance(anomalies, pd.Series)
    assert anomalies.sum() >= 1


def test_detect_isolation_forest(anomaly_df):
    anomalies = detect_isolation_forest_anomalies(anomaly_df)
    assert isinstance(anomalies, pd.Series)
    assert anomalies.dtype == bool


def test_combine_anomaly_flags(anomaly_df):
    df = anomaly_df.copy()
    df["anomaly_zscore"] = detect_zscore_anomalies(df["temperature"])
    df["anomaly_iforest"] = detect_isolation_forest_anomalies(df)
    result = combine_anomaly_flags(df)
    assert "anomaly_combined" in result.columns
    assert result["anomaly_combined"].dtype == bool
