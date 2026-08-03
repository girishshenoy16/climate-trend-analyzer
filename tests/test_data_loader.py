"""Tests for data loader module."""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import (
    validate_schema,
    validate_date_bounds,
    convert_types,
    compute_data_quality_report,
)


@pytest.fixture
def sample_df():
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    return pd.DataFrame({
        "date": dates,
        "temperature": np.random.uniform(15, 35, len(dates)),
        "temperature_max": np.random.uniform(20, 40, len(dates)),
        "temperature_min": np.random.uniform(10, 30, len(dates)),
        "precipitation": np.random.uniform(0, 10, len(dates)),
        "humidity": np.random.uniform(30, 90, len(dates)),
        "solar_radiation": np.random.uniform(10, 25, len(dates)),
        "wind_speed": np.random.uniform(0, 15, len(dates)),
    })


def test_validate_schema_valid(sample_df):
    issues = validate_schema(sample_df)
    assert len(issues["missing_columns"]) == 0


def test_validate_schema_missing_column():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"])})
    issues = validate_schema(df)
    assert "temperature" in issues["missing_columns"]


def test_validate_date_bounds(sample_df):
    assert validate_date_bounds(sample_df) is True


def test_convert_types(sample_df):
    df = convert_types(sample_df)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_float_dtype(df["temperature"])


def test_compute_data_quality_report(sample_df):
    report = compute_data_quality_report(sample_df)
    assert "total_rows" in report
    assert "missing_percentage" in report
    assert report["total_rows"] == len(sample_df)
