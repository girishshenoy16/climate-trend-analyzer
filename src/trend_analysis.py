"""
Long-term Trend & Seasonal Analysis for Climate Trend Analyzer.

Implements linear trend estimation (y = mx + c) using scipy.stats.linregress,
warming rate calculation, and STL (Seasonal-Trend decomposition using Loess) decomposition.
Includes statistical significance validation for all computed trends.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL

from src.constants import (
    COL_DATE,
    COL_PRECIPITATION,
    COL_RESIDUAL,
    COL_SEASONAL,
    COL_TEMP_MEAN,
    COL_TREND,
)
from src.logger import get_logger, check_trend_significance

logger = get_logger("trend_analysis")


def compute_linear_trend(series: pd.Series, dates: pd.Series) -> dict[str, Any]:
    """
    Compute linear trend (y = mx + c) for a time series using scipy.stats.linregress.

    Args:
        series: Numeric time series.
        dates: Corresponding dates.

    Returns:
        Dictionary with slope, intercept, r-value, p-value, std_err,
        r_squared, and warming_rate_per_decade.
    """
    date_numeric = (dates - dates.min()).dt.total_seconds() / 86400

    mask = series.notna() & date_numeric.notna()
    x = date_numeric[mask].values
    y = series[mask].values

    if len(x) < 2:
        return {
            "slope": 0.0, "intercept": 0.0, "r_value": 0.0,
            "r_squared": 0.0, "p_value": 1.0, "std_err": 0.0,
            "warming_rate_per_decade": 0.0,
        }

    # Linear regression via scipy
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Convert slope (°C/day) to °C/decade
    warming_rate_per_decade = slope * 365.25 * 10

    result = {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_value": float(r_value),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "warming_rate_per_decade": float(warming_rate_per_decade),
    }

    # Log significance warnings
    check_trend_significance(p_value, r_value ** 2, "Temperature")

    return result


def compute_trend_component(series: pd.Series, dates: pd.Series) -> np.ndarray:
    """
    Compute the linear trend component for a time series.

    Args:
        series: Numeric time series.
        dates: Corresponding dates.

    Returns:
        Array of trend values.
    """
    date_numeric = (dates - dates.min()).dt.total_seconds() / 86400
    mask = series.notna() & date_numeric.notna()

    x = date_numeric[mask].values
    y = series[mask].values

    if len(x) < 2:
        return np.full(len(series), series.mean())

    slope, intercept, _, _, _ = stats.linregress(x, y)
    trend_values = slope * date_numeric.values + intercept

    return trend_values


def perform_stl_decomposition(
    series: pd.Series,
    period: int = 365,
    seasonal: int = 7,
    robust: bool = True,
) -> dict[str, pd.Series]:
    """
    Perform STL (Seasonal-Trend decomposition using Loess) decomposition.

    Args:
        series: Time series to decompose.
        period: Seasonal period (default 365 for daily data).
        seasonal: Seasonal smoothing parameter.
        robust: Whether to use robust fitting.

    Returns:
        Dictionary with trend, seasonal, and residual components.
    """
    clean_series = series.dropna()

    if len(clean_series) < 2 * period:
        logger.warning(
            f"Series length ({len(clean_series)}) < 2 * period ({2 * period}). "
            "Reducing period for valid decomposition."
        )
        period = max(7, len(clean_series) // 3)

    stl = STL(
        clean_series,
        period=period,
        seasonal=seasonal,
        robust=robust,
    )

    result = stl.fit()

    components = {
        COL_TREND: result.trend,
        COL_SEASONAL: result.seasonal,
        COL_RESIDUAL: result.resid,
    }

    logger.info(f"STL decomposition complete (period={period})")
    logger.info(f"  Trend range: {result.trend.min():.2f} to {result.trend.max():.2f}")
    logger.info(f"  Seasonal range: {result.seasonal.min():.2f} to {result.seasonal.max():.2f}")

    return components


def analyze_temperature_trends(df: pd.DataFrame) -> dict[str, Any]:
    """
    Comprehensive temperature trend analysis.

    Args:
        df: Processed DataFrame.

    Returns:
        Dictionary with trend analysis results.
    """
    results = {}

    if COL_TEMP_MEAN not in df.columns:
        logger.error("Temperature column not found")
        return results

    # Linear trend analysis
    trend = compute_linear_trend(df[COL_TEMP_MEAN], df[COL_DATE])
    results["linear_trend"] = trend

    logger.info(
        f"Temperature linear trend: {trend['warming_rate_per_decade']:.3f} °C/decade "
        f"(R²={trend['r_squared']:.4f}, p={trend['p_value']:.4e})"
    )

    # STL decomposition (only if sufficient data)
    if len(df) >= 730:
        stl_result = perform_stl_decomposition(df[COL_TEMP_MEAN])
        results["stl_components"] = stl_result

    # Trend line values for plotting
    results["trend_line"] = compute_trend_component(df[COL_TEMP_MEAN], df[COL_DATE])

    return results


def analyze_precipitation_trends(df: pd.DataFrame) -> dict[str, Any]:
    """
    Precipitation trend analysis with significance checking.

    Args:
        df: Processed DataFrame.

    Returns:
        Dictionary with precipitation trend results.
    """
    results = {}

    if COL_PRECIPITATION not in df.columns:
        return results

    # Linear trend
    trend = compute_linear_trend(df[COL_PRECIPITATION], df[COL_DATE])
    results["linear_trend"] = trend

    logger.info(
        f"Precipitation trend: {trend['slope']:.6f} mm/day per day "
        f"(p={trend['p_value']:.4e})"
    )

    # Check significance
    check_trend_significance(
        trend["p_value"], trend["r_squared"], "Precipitation"
    )

    results["trend_line"] = compute_trend_component(df[COL_PRECIPITATION], df[COL_DATE])

    return results


def run_trend_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """
    Execute the complete trend and seasonal analysis pipeline.

    Args:
        df: Processed DataFrame.

    Returns:
        Dictionary with all trend analysis results.
    """
    results = {
        "temperature": analyze_temperature_trends(df),
        "precipitation": analyze_precipitation_trends(df),
    }

    logger.info("Trend analysis pipeline complete")
    return results
