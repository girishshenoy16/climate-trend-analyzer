"""Tests for trend analysis module."""

import pytest
import pandas as pd
import numpy as np
from src.trend_analysis import compute_linear_trend, perform_stl_decomposition


@pytest.fixture
def trend_df():
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", "2024-12-31", freq="D")
    n = len(dates)
    # Strong upward trend: ~2.5°C/decade to clearly exceed noise
    trend = 0.7 * np.arange(n) / 365.25
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    noise = np.random.normal(0, 1, n)
    temps = 25 + trend + seasonal + noise
    return pd.DataFrame({"date": dates, "temperature": temps})


def test_compute_linear_trend(trend_df):
    result = compute_linear_trend(trend_df["temperature"], trend_df["date"])
    assert "slope" in result
    assert "r_squared" in result
    assert "warming_rate_per_decade" in result
    assert result["warming_rate_per_decade"] > 0


def test_perform_stl_decomposition(trend_df):
    components = perform_stl_decomposition(trend_df["temperature"], period=365)
    assert "trend" in components
    assert "seasonal_component" in components
    assert "residual" in components
    assert len(components["trend"]) > 0
