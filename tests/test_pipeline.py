"""End-to-end pipeline integration test."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.synthetic_generator import generate_synthetic_data
from src.preprocessing import preprocess_pipeline
from src.eda import run_eda
from src.trend_analysis import run_trend_analysis
from src.anomaly_detection import run_anomaly_detection
from src.forecasting import run_forecasting
from src.report_generator import generate_executive_summary


@pytest.fixture
def synthetic_df():
    return generate_synthetic_data(start_year=2015, end_year=2020, seed=42)


def test_full_pipeline(synthetic_df):
    """Test complete pipeline from raw data to executive summary."""
    # Preprocessing
    df = preprocess_pipeline(synthetic_df)
    assert len(df) > 0
    assert "year" in df.columns
    assert "season" in df.columns

    # EDA
    eda = run_eda(df)
    assert "summary_statistics" in eda
    assert "correlation_matrix" in eda

    # Trend analysis
    trends = run_trend_analysis(df)
    assert "temperature" in trends

    # Anomaly detection
    df, anomaly_summary = run_anomaly_detection(df)
    assert "anomaly_combined" in df.columns
    assert "total_anomaly_days" in anomaly_summary

    # Forecasting
    forecast = run_forecasting(df)
    assert "forecast_df" in forecast
    assert "model_metrics" in forecast

    # Executive summary
    summary = generate_executive_summary(df, trends, anomaly_summary, forecast, eda)
    assert "kpis" in summary
    assert "risk_score" in summary
    assert "insights" in summary
    assert summary["kpis"]["risk_category"] in ["Low", "Moderate", "High", "Very High"]
