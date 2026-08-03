"""
Exploratory Data Analysis Engine for Climate Trend Analyzer.

Computes summary statistics, monthly and annual aggregations,
and Pearson correlation matrix across climate variables.
"""

from typing import Any

import pandas as pd
import numpy as np

from src.constants import (
    COL_DATE,
    COL_HUMIDITY,
    COL_MONTH,
    COL_PRECIPITATION,
    COL_SOLAR_RADIATION,
    COL_TEMP_MEAN,
    COL_WIND_SPEED,
    COL_YEAR,
    CLIMATE_VARIABLES,
)
from src.logger import get_logger, log_pipeline_stage
from src.utils import compute_summary_stats

logger = get_logger("eda")

# Variables suitable for correlation analysis
CORRELATION_VARIABLES = [
    COL_TEMP_MEAN,
    COL_PRECIPITATION,
    COL_HUMIDITY,
    COL_SOLAR_RADIATION,
    COL_WIND_SPEED,
]


def compute_summary_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute overall summary statistics for all climate variables.

    Args:
        df: Processed DataFrame.

    Returns:
        Dictionary of summary statistics per variable.
    """
    stats = {}
    for col in CLIMATE_VARIABLES:
        if col in df.columns:
            stats[col] = compute_summary_stats(df[col])

    logger.info(f"Summary statistics computed for {len(stats)} variables")
    return stats


def compute_monthly_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly mean aggregations for climate variables.

    Args:
        df: Processed DataFrame with year and month columns.

    Returns:
        DataFrame with monthly aggregated values.
    """
    if COL_YEAR not in df.columns or COL_MONTH not in df.columns:
        df = df.copy()
        df[COL_YEAR] = df[COL_DATE].dt.year
        df[COL_MONTH] = df[COL_DATE].dt.month

    numeric_cols = [c for c in CLIMATE_VARIABLES if c in df.columns]

    monthly = (
        df.groupby([COL_YEAR, COL_MONTH])[numeric_cols]
        .mean()
        .reset_index()
    )

    logger.info(f"Monthly aggregations: {len(monthly)} month-year records")
    return monthly


def compute_annual_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annual mean aggregations for climate variables.

    Args:
        df: Processed DataFrame.

    Returns:
        DataFrame with annual aggregated values.
    """
    if COL_YEAR not in df.columns:
        df = df.copy()
        df[COL_YEAR] = df[COL_DATE].dt.year

    numeric_cols = [c for c in CLIMATE_VARIABLES if c in df.columns]

    annual = (
        df.groupby(COL_YEAR)[numeric_cols]
        .agg(["mean", "std", "min", "max"])
    )

    # Flatten column names
    annual.columns = [f"{col}_{stat}" for col, stat in annual.columns]
    annual = annual.reset_index()

    logger.info(f"Annual aggregations: {len(annual)} year records")
    return annual


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix across climate variables.

    Args:
        df: Processed DataFrame.

    Returns:
        Correlation matrix DataFrame.
    """
    available_cols = [c for c in CORRELATION_VARIABLES if c in df.columns]

    if len(available_cols) < 2:
        logger.warning("Insufficient variables for correlation analysis")
        return pd.DataFrame()

    corr_matrix = df[available_cols].corr(method="pearson")

    logger.info(f"Correlation matrix computed: {corr_matrix.shape}")
    return corr_matrix


def compute_annual_trends(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute annual warming and precipitation trends.

    Returns:
        Dictionary with trend metrics.
    """
    trends = {}

    if COL_YEAR not in df.columns:
        df = df.copy()
        df[COL_YEAR] = df[COL_DATE].dt.year

    # Temperature trend
    if COL_TEMP_MEAN in df.columns:
        annual_temp = df.groupby(COL_YEAR)[COL_TEMP_MEAN].mean()
        if len(annual_temp) >= 2:
            x = np.arange(len(annual_temp))
            slope, intercept = np.polyfit(x, annual_temp.values, 1)
            trends["temperature_trend_per_year"] = float(slope)
            trends["temperature_trend_per_decade"] = float(slope * 10)

    # Precipitation trend
    if COL_PRECIPITATION in df.columns:
        annual_precip = df.groupby(COL_YEAR)[COL_PRECIPITATION].mean()
        if len(annual_precip) >= 2:
            x = np.arange(len(annual_precip))
            slope, intercept = np.polyfit(x, annual_precip.values, 1)
            trends["precipitation_trend_per_year"] = float(slope)

    logger.info(f"Annual trends computed: {len(trends)} metrics")
    return trends


def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    """
    Execute the complete EDA pipeline.

    Args:
        df: Processed DataFrame.

    Returns:
        Dictionary with all EDA results.
    """
    log_pipeline_stage("Exploratory Data Analysis")

    results = {
        "summary_statistics": compute_summary_statistics(df),
        "monthly_aggregations": compute_monthly_aggregations(df),
        "annual_aggregations": compute_annual_aggregations(df),
        "correlation_matrix": compute_correlation_matrix(df),
        "annual_trends": compute_annual_trends(df),
    }

    logger.info("EDA pipeline complete")
    return results
