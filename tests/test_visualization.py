"""Tests for visualization module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.visualization import (
    plot_temperature_trend,
    plot_rainfall_trend,
    plot_correlation_heatmap,
    plot_monthly_distribution,
)


@pytest.fixture
def viz_df():
    np.random.seed(42)
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
        "month": dates.month,
    })


def test_plot_temperature_trend(viz_df):
    path = plot_temperature_trend(viz_df)
    assert path.exists()
    assert path.suffix == ".png"


def test_plot_rainfall_trend(viz_df):
    path = plot_rainfall_trend(viz_df)
    assert path.exists()


def test_plot_correlation_heatmap(viz_df):
    corr = viz_df[["temperature", "precipitation", "humidity"]].corr()
    path = plot_correlation_heatmap(corr)
    assert path.exists()


def test_plot_monthly_distribution(viz_df):
    path = plot_monthly_distribution(viz_df)
    assert path.exists()
