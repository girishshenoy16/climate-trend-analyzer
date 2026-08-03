"""Tests for preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    impute_missing_values,
    add_temporal_features,
    add_rolling_averages,
    add_lag_features,
)


@pytest.fixture
def sample_df():
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    temps = np.random.uniform(15, 35, len(dates))
    temps[10] = np.nan
    temps[20] = np.nan
    return pd.DataFrame({
        "date": dates,
        "temperature": temps,
        "precipitation": np.random.uniform(0, 10, len(dates)),
    })


def test_impute_missing_values(sample_df):
    assert sample_df["temperature"].isna().sum() > 0
    result = impute_missing_values(sample_df)
    assert result["temperature"].isna().sum() == 0


def test_add_temporal_features(sample_df):
    result = add_temporal_features(sample_df)
    assert "year" in result.columns
    assert "month" in result.columns
    assert "day_of_year" in result.columns
    assert "season" in result.columns


def test_add_rolling_averages(sample_df):
    result = add_rolling_averages(sample_df)
    assert "temperature_ma7" in result.columns
    assert "temperature_ma30" in result.columns
    assert "temperature_ma365" in result.columns


def test_add_lag_features(sample_df):
    result = add_lag_features(sample_df)
    assert "temperature_lag1" in result.columns
    assert "temperature_lag7" in result.columns
    assert "temperature_lag30" in result.columns
    assert result["temperature_lag1"].isna().sum() == 0
