"""Tests for forecasting module."""

import pytest
import pandas as pd
import numpy as np
from src.forecasting import (
    fit_holt_winters,
    evaluate_model,
    generate_forecast,
    validate_forecast_quality,
)
from src.constants import COL_DATE, COL_FORECAST, COL_FORECAST_UPPER, COL_FORECAST_LOWER


@pytest.fixture
def forecast_df():
    dates = pd.date_range("2015-01-01", "2024-12-31", freq="D")
    n = len(dates)
    trend = 0.03 * np.arange(n) / 365.25
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    temps = 25 + trend + seasonal + np.random.normal(0, 2, n)
    return pd.DataFrame({"date": dates, "temperature": temps})


@pytest.fixture
def valid_forecast_df():
    dates = pd.date_range("2025-01-01", periods=365, freq="D")
    temps = 25 + 0.03 * np.arange(365) / 365.25 + 10 * np.sin(2 * np.pi * np.arange(365) / 365.25)
    return pd.DataFrame({
        COL_DATE: dates,
        COL_FORECAST: temps,
        COL_FORECAST_UPPER: temps + 2.0,
        COL_FORECAST_LOWER: temps - 2.0,
    })


@pytest.fixture
def historical_series():
    dates = pd.date_range("2015-01-01", "2024-12-31", freq="D")
    n = len(dates)
    trend = 0.03 * np.arange(n) / 365.25
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    temps = 25 + trend + seasonal + np.random.normal(0, 2, n)
    return pd.Series(temps, index=dates)


def test_evaluate_model():
    actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = pd.Series([1.1, 2.1, 2.9, 4.2, 4.8])
    metrics = evaluate_model(actual, predicted)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "mape" in metrics
    assert metrics["rmse"] > 0


def test_fit_holt_winters(forecast_df):
    series = forecast_df.set_index("date")["temperature"]
    series = series.asfreq("D")
    model = fit_holt_winters(series, seasonal_periods=365)
    assert model is not None
    assert hasattr(model, "fittedvalues")


def test_generate_forecast(forecast_df):
    results = generate_forecast(forecast_df, horizon_years=1)
    assert "forecast_df" in results
    assert "model_metrics" in results
    assert "trend_per_year" in results
    assert len(results["forecast_df"]) > 0


def test_validate_forecast_quality_pass(valid_forecast_df, historical_series):
    trend_info = {"trend_per_decade": 0.3}
    hist_trend = 0.3
    result = validate_forecast_quality(valid_forecast_df, historical_series, trend_info, hist_trend)
    assert result["passed"] is True
    assert len(result["issues"]) == 0
    assert result["details"]["nan_count"] == 0
    assert result["details"]["invalid_ci_count"] == 0


def test_validate_forecast_quality_fail_nan(valid_forecast_df, historical_series):
    valid_forecast_df.loc[valid_forecast_df.index[0], COL_FORECAST] = np.nan
    trend_info = {"trend_per_decade": 0.3}
    hist_trend = 0.3
    result = validate_forecast_quality(valid_forecast_df, historical_series, trend_info, hist_trend)
    assert result["passed"] is False
    assert any("NaN" in issue for issue in result["issues"])
    assert result["details"]["nan_count"] == 1


def test_validate_forecast_quality_fail_slope(valid_forecast_df, historical_series):
    trend_info = {"trend_per_decade": 15.0}
    hist_trend = 0.3
    result = validate_forecast_quality(valid_forecast_df, historical_series, trend_info, hist_trend)
    assert result["passed"] is False
    assert any("extreme" in issue.lower() for issue in result["issues"])
    assert result["details"]["forecast_trend_per_decade"] == 15.0


def test_validate_forecast_quality_fail_ci(valid_forecast_df, historical_series):
    valid_forecast_df.loc[valid_forecast_df.index[0], COL_FORECAST_UPPER] = valid_forecast_df.loc[valid_forecast_df.index[0], COL_FORECAST_LOWER] - 5.0
    trend_info = {"trend_per_decade": 0.3}
    hist_trend = 0.3
    result = validate_forecast_quality(valid_forecast_df, historical_series, trend_info, hist_trend)
    assert result["passed"] is False
    assert any("invalid confidence intervals" in issue.lower() for issue in result["issues"])
    assert result["details"]["invalid_ci_count"] >= 1


def test_validate_forecast_quality_fail_consistency(valid_forecast_df, historical_series):
    valid_forecast_df[COL_FORECAST] = valid_forecast_df[COL_FORECAST] + 100.0
    trend_info = {"trend_per_decade": 0.3}
    hist_trend = 0.3
    result = validate_forecast_quality(valid_forecast_df, historical_series, trend_info, hist_trend)
    assert result["passed"] is False
    assert any("deviates" in issue.lower() for issue in result["issues"])
    assert result["details"]["mean_deviation_pct"] > 20.0


def test_validate_forecast_quality_fail_values(valid_forecast_df, historical_series):
    valid_forecast_df[COL_FORECAST] = 100.0
    trend_info = {"trend_per_decade": 0.3}
    hist_trend = 0.3
    result = validate_forecast_quality(valid_forecast_df, historical_series, trend_info, hist_trend)
    assert result["passed"] is False
    assert any("physically unreasonable" in issue.lower() for issue in result["issues"])
    assert result["details"]["forecast_max"] == 100.0
