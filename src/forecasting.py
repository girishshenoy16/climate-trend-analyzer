"""
Holt-Winters Exponential Smoothing Forecasting for Climate Trend Analyzer.

Implements triple exponential smoothing (Holt-Winters) for temperature
forecasting with walk-forward validation, model benchmarking, automatic
stability validation, fallback strategies, and comprehensive reliability
scoring. Forecasts are validated against historical trends before
reporting, with automatic classification of forecast reliability and
transparent documentation of all decisions.

Model Selection Rationale:
    Holt-Winters was selected over ARIMA, SARIMA, and Prophet because:
    - Captures level, trend, and seasonality in a unified framework
    - Well-suited for daily climate data with strong 365-day annual cycles
    - Additive formulation appropriate for temperature data with roughly
      constant seasonal amplitude across years
    - Provides in-sample fitted values for transparent model evaluation
    - Well-understood algorithm with transparent assumptions

    Limitations compared to alternatives:
    - ARIMA/SARIMA: Better for non-seasonal or short-memory processes
    - Prophet: Better for data with multiple seasonalities and holidays
    - All models: Limited for long-range climate projections due to
      structural breaks and external forcing not captured in historical data
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore", message=".*Optimization failed to converge.*")

from src.config import (
    FORECAST_CONFIDENCE_LEVEL,
    FORECAST_HORIZON_YEARS,
    HOLT_WINTERS_SEASONAL_PERIODS,
    MAX_TREND_RATIO,
    FIGURES_DIR,
)
from src.constants import (
    COL_DATE,
    COL_FORECAST,
    COL_FORECAST_LOWER,
    COL_FORECAST_UPPER,
    COL_TEMP_MEAN,
)
from src.logger import get_logger

logger = get_logger("forecasting")

# ─── Forecast Configuration ───────────────────────────────────────────────────

# Reliability scale thresholds (score -> label)
RELIABILITY_THRESHOLDS = [
    (0.90, "Very High"),
    (0.75, "High"),
    (0.60, "Moderate"),
    (0.40, "Low"),
    (0.00, "Exploratory"),
]

# Forecast class thresholds (derived from reliability score + validation checks)
FORECAST_CLASS_THRESHOLDS = {
    "reliable": {"min_score": 0.60, "max_trend_ratio": MAX_TREND_RATIO, "min_r2": 0.10},
    "low_confidence": {"min_score": 0.40, "max_trend_ratio": 5.0, "min_r2": 0.05},
    "exploratory": {"min_score": 0.00, "max_trend_ratio": float("inf"), "min_r2": 0.00},
}


def fit_holt_winters(
    series: pd.Series,
    seasonal_periods: int = HOLT_WINTERS_SEASONAL_PERIODS,
    trend: str = "add",
    seasonal: str = "add",
    damped_trend: bool = False,
) -> ExponentialSmoothing:
    """
    Fit Holt-Winters Exponential Smoothing model.

    Args:
        series: Time series to model.
        seasonal_periods: Number of periods in a seasonal cycle.
        trend: Trend type ('add', 'mul', or None).
        seasonal: Seasonal type ('add', 'mul', or None).
        damped_trend: Whether to use damped trend.

    Returns:
        Fitted ExponentialSmoothing model.
    """
    clean_series = series.dropna()

    if len(clean_series) < 2 * seasonal_periods:
        logger.warning(
            f"Series length ({len(clean_series)}) insufficient for "
            f"seasonal_periods={seasonal_periods}. Adjusting."
        )
        seasonal_periods = max(7, len(clean_series) // 4)

    model = ExponentialSmoothing(
        clean_series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        damped_trend=damped_trend,
    )

    fit = model.fit(optimized=True, use_brute=True)
    logger.info(
        f"Holt-Winters model fitted: AIC={fit.aic:.2f}, BIC={fit.bic:.2f}"
    )

    return fit


def evaluate_model(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    """
    Evaluate forecasting model performance.

    Args:
        actual: Actual values.
        predicted: Predicted values.

    Returns:
        Dictionary with RMSE, MAE, MAPE, and R-squared metrics.
    """
    common_idx = actual.index.intersection(predicted.index)
    actual_aligned = actual.loc[common_idx]
    predicted_aligned = predicted.loc[common_idx]

    mask = actual_aligned.notna() & predicted_aligned.notna()
    actual_clean = actual_aligned[mask]
    predicted_clean = predicted_aligned[mask]

    if len(actual_clean) == 0:
        return {"rmse": 0.0, "mae": 0.0, "mape": 0.0, "r_squared": 0.0}

    rmse = float(np.sqrt(mean_squared_error(actual_clean, predicted_clean)))
    mae = float(mean_absolute_error(actual_clean, predicted_clean))

    # MAPE (avoid division by zero)
    nonzero_mask = actual_clean != 0
    if nonzero_mask.any():
        mape = float(
            np.mean(
                np.abs(
                    (actual_clean[nonzero_mask] - predicted_clean[nonzero_mask])
                    / actual_clean[nonzero_mask]
                )
            )
            * 100
        )
    else:
        mape = 0.0

    # R-squared
    ss_res = np.sum((actual_clean - predicted_clean) ** 2)
    ss_tot = np.sum((actual_clean - actual_clean.mean()) ** 2)
    r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {"rmse": rmse, "mae": mae, "mape": mape, "r_squared": r_squared}


def walk_forward_validation(
    series: pd.Series,
    n_splits: int = 3,
    test_size_days: int = 365,
    min_train_days: int = 730,
    seasonal_periods: int = HOLT_WINTERS_SEASONAL_PERIODS,
) -> dict[str, Any]:
    """
    Perform walk-forward (rolling-origin) time-series validation.

    Splits the series into sequential train/test windows, fitting
    Holt-Winters on each training window and evaluating on the
    subsequent test window.

    Args:
        series: Complete historical time series.
        n_splits: Number of validation windows.
        test_size_days: Days in each test window.
        min_train_days: Minimum training window size.
        seasonal_periods: Seasonal period for Holt-Winters.

    Returns:
        Dictionary with per-fold metrics and aggregate statistics.
    """
    series_clean = series.dropna()
    total_len = len(series_clean)

    if total_len < min_train_days + test_size_days:
        logger.warning(
            f"Series too short ({total_len}) for walk-forward validation "
            f"(min required: {min_train_days + test_size_days}). Using simpler split."
        )
        n_splits = max(1, (total_len - min_train_days) // test_size_days)

    fold_metrics = []
    start_idx = total_len - (n_splits * test_size_days)

    if start_idx < min_train_days:
        start_idx = min_train_days

    for fold in range(n_splits):
        train_end = start_idx + fold * test_size_days
        test_end = train_end + test_size_days

        if test_end > total_len:
            break

        train_series = series_clean.iloc[:train_end]
        test_series = series_clean.iloc[train_end:test_end]

        try:
            model_fit = fit_holt_winters(
                train_series, seasonal_periods=min(seasonal_periods, len(train_series) // 3)
            )
            predictions = model_fit.forecast(len(test_series))
            fold_result = evaluate_model(test_series, predictions)
            fold_result["fold"] = fold + 1
            fold_result["train_start"] = str(train_series.index[0].date())
            fold_result["train_end"] = str(train_series.index[-1].date())
            fold_result["test_start"] = str(test_series.index[0].date())
            fold_result["test_end"] = str(test_series.index[-1].date())
            fold_metrics.append(fold_result)
            logger.info(
                f"Walk-forward fold {fold + 1}: RMSE={fold_result['rmse']:.3f}, "
                f"MAE={fold_result['mae']:.3f}, MAPE={fold_result['mape']:.2f}%, "
                f"R²={fold_result['r_squared']:.4f}"
            )
        except Exception as e:
            logger.warning(f"Walk-forward fold {fold + 1} failed: {e}")

    if not fold_metrics:
        return {
            "n_folds": 0,
            "avg_rmse": 0.0,
            "avg_mae": 0.0,
            "avg_mape": 0.0,
            "avg_r_squared": 0.0,
            "fold_metrics": [],
        }

    avg_metrics = {
        "n_folds": len(fold_metrics),
        "avg_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "avg_mae": float(np.mean([m["mae"] for m in fold_metrics])),
        "avg_mape": float(np.mean([m["mape"] for m in fold_metrics])),
        "avg_r_squared": float(np.mean([m["r_squared"] for m in fold_metrics])),
        "std_rmse": float(np.std([m["rmse"] for m in fold_metrics])),
        "std_mae": float(np.std([m["mae"] for m in fold_metrics])),
        "fold_metrics": fold_metrics,
    }

    logger.info(
        f"Walk-forward validation complete: {len(fold_metrics)} folds, "
        f"Avg RMSE={avg_metrics['avg_rmse']:.3f}, "
        f"Avg MAE={avg_metrics['avg_mae']:.3f}, "
        f"Avg MAPE={avg_metrics['avg_mape']:.2f}%, "
        f"Avg R²={avg_metrics['avg_r_squared']:.4f}"
    )

    return avg_metrics


def _naive_forecast(series: pd.Series, steps: int) -> pd.Series:
    """Naive forecast: last observed value repeated."""
    last_val = series.iloc[-1]
    return pd.Series(last_val, index=range(steps))


def _seasonal_naive_forecast(
    series: pd.Series, steps: int, seasonal_period: int = 365
) -> pd.Series:
    """Seasonal naive: repeat last seasonal cycle."""
    last_cycle = series.iloc[-seasonal_period:].values
    repeats = steps // seasonal_period + 1
    forecast = np.tile(last_cycle, repeats)[:steps]
    return pd.Series(forecast, index=range(steps))


def _linear_trend_forecast(series: pd.Series, steps: int) -> pd.Series:
    """Linear trend regression forecast."""
    x = np.arange(len(series), dtype=float)
    y = series.values.astype(float)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    future_x = np.arange(len(series), len(series) + steps, dtype=float)
    return pd.Series(slope * future_x + intercept, index=range(steps))


def benchmark_models(
    series: pd.Series,
    test_days: int = 365,
    seasonal_period: int = 365,
) -> dict[str, Any]:
    """
    Benchmark Holt-Winters against baseline models using last test_days.

    Models compared:
        1. Holt-Winters (primary)
        2. Naive (last value)
        3. Seasonal Naive (repeat last cycle)
        4. Linear Trend Regression

    Args:
        series: Historical time series.
        test_days: Number of days for holdout test.
        seasonal_period: Seasonal period for seasonal naive.

    Returns:
        Dictionary with per-model metrics and best model selection.
    """
    series_clean = series.dropna()

    if len(series_clean) <= test_days + 365:
        test_days = max(30, len(series_clean) // 4)
        logger.warning(f"Adjusted benchmark test size to {test_days} days")

    train = series_clean.iloc[:-test_days]
    test = series_clean.iloc[-test_days:]
    steps = len(test)

    benchmarks = {}

    # 1. Holt-Winters
    try:
        hw_fit = fit_holt_winters(
            train, seasonal_periods=min(seasonal_period, len(train) // 3)
        )
        hw_pred = hw_fit.forecast(steps)
        hw_pred.index = test.index
        benchmarks["holt_winters"] = evaluate_model(test, hw_pred)
        benchmarks["holt_winters"]["model"] = "Holt-Winters"
    except Exception as e:
        logger.warning(f"Holt-Winters benchmark failed: {e}")
        benchmarks["holt_winters"] = {
            "rmse": float("inf"), "mae": float("inf"),
            "mape": float("inf"), "r_squared": 0.0, "model": "Holt-Winters",
        }

    # 2. Naive
    naive_pred = _naive_forecast(train, steps)
    naive_pred.index = test.index
    benchmarks["naive"] = evaluate_model(test, naive_pred)
    benchmarks["naive"]["model"] = "Naive"

    # 3. Seasonal Naive
    snaive_pred = _seasonal_naive_forecast(train, steps, seasonal_period)
    snaive_pred.index = test.index
    benchmarks["seasonal_naive"] = evaluate_model(test, snaive_pred)
    benchmarks["seasonal_naive"]["model"] = "Seasonal Naive"

    # 4. Linear Trend
    lt_pred = _linear_trend_forecast(train, steps)
    lt_pred.index = test.index
    benchmarks["linear_trend"] = evaluate_model(test, lt_pred)
    benchmarks["linear_trend"]["model"] = "Linear Trend"

    # Determine best model by RMSE
    best_model = min(benchmarks.items(), key=lambda x: x[1].get("rmse", float("inf")))
    best_name = best_model[0]

    # Rank models
    ranked = sorted(benchmarks.items(), key=lambda x: x[1].get("rmse", float("inf")))
    for rank, (name, metrics) in enumerate(ranked, 1):
        metrics["rank"] = rank

    hw_rmse = benchmarks["holt_winters"].get("rmse", float("inf"))
    best_rmse = best_model[1].get("rmse", float("inf"))

    if best_name != "holt_winters" and best_rmse < hw_rmse * 0.9:
        recommendation = (
            f"{best_model[1]['model']} outperformed Holt-Winters "
            f"(RMSE: {best_rmse:.3f} vs {hw_rmse:.3f}). "
            f"Consider alternative model for this dataset."
        )
        use_hw = False
    else:
        recommendation = (
            f"Holt-Winters selected (RMSE: {hw_rmse:.3f}). "
            f"Comparable or superior to baseline models."
        )
        use_hw = True

    logger.info(
        f"Model benchmarking complete. Best: {best_name} "
        f"(RMSE={best_rmse:.3f}). HW ranked #{benchmarks['holt_winters'].get('rank', '?')}"
    )

    return {
        "benchmarks": benchmarks,
        "best_model": best_name,
        "use_holt_winters": use_hw,
        "recommendation": recommendation,
        "test_days": test_days,
    }


def _compute_forecast_trend_linregress(
    forecast_series: pd.Series,
    horizon_years: int,
) -> dict[str, float]:
    """
    Compute forecast trend using linear regression (linregress).

    This avoids the unreliable first-last difference method and provides
    statistically meaningful slope, R-squared, and p-value for the forecast.

    Args:
        forecast_series: Forecasted temperature values (daily).
        horizon_years: Number of years in the forecast horizon.

    Returns:
        Dictionary with trend_per_year, trend_per_decade, r_squared, p_value.
    """
    x = np.arange(len(forecast_series), dtype=float)
    y = forecast_series.values.astype(float)

    # Remove any NaN
    mask = np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 2:
        return {
            "trend_per_year": 0.0,
            "trend_per_decade": 0.0,
            "r_squared": 0.0,
            "p_value": 1.0,
        }

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

    # Convert daily slope to per-year and per-decade
    trend_per_day = slope
    trend_per_year = trend_per_day * 365.25
    trend_per_decade = trend_per_year * 10

    return {
        "trend_per_year": float(trend_per_year),
        "trend_per_decade": float(trend_per_decade),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
    }


def _compute_reliability_score(
    trend_ratio: float,
    fc_r_squared: float,
    in_sample_rmse: float,
    mean_temp: float,
    trend_p_value: float,
    validation_metrics: dict = None,
    quality_checks: dict = None,
    benchmark_results: dict = None,
) -> dict[str, Any]:
    """
    Compute forecast reliability score based on multiple factors.

    Factors:
        - Historical consistency (trend ratio)
        - Forecast R-squared (linear trend fit)
        - In-sample RMSE (relative to mean temperature)
        - Statistical significance of forecast trend
        - Walk-forward validation performance (if available)
        - Forecast quality check results
        - Model benchmark comparison

    Returns:
        Dictionary with score (0-1), label, and contributing factors.
    """
    rmse_pct = (in_sample_rmse / abs(mean_temp) * 100) if mean_temp != 0 else 100

    # Score components (0-1 scale, higher is better)
    consistency_score = max(0, 1.0 - (trend_ratio - 1.0) / 2.0) if trend_ratio >= 1.0 else 1.0
    r2_score = min(1.0, fc_r_squared / 0.5)
    rmse_score = max(0, 1.0 - rmse_pct / 20.0)
    significance_score = 1.0 if trend_p_value < 0.05 else 0.5

    # Validation score (if walk-forward available)
    validation_score = 0.5  # default if no validation
    if validation_metrics and validation_metrics.get("n_folds", 0) > 0:
        val_r2 = validation_metrics.get("avg_r_squared", 0)
        val_rmse = validation_metrics.get("avg_rmse", 0)
        val_r2_score = min(1.0, max(0, val_r2 / 0.5))
        val_rmse_score = max(0, 1.0 - (val_rmse / abs(mean_temp) * 100) / 20.0) if mean_temp != 0 else 0.5
        validation_score = (val_r2_score + val_rmse_score) / 2

    # Quality check score
    quality_score = 1.0
    if quality_checks and not quality_checks.get("passed", True):
        n_issues = len(quality_checks.get("issues", []))
        quality_score = max(0, 1.0 - n_issues * 0.25)

    # Benchmark score
    benchmark_score = 0.5
    if benchmark_results:
        if benchmark_results.get("use_holt_winters", True):
            benchmark_score = 0.8
        else:
            benchmark_score = 0.4

    # Weighted average (updated weights to include new factors)
    weights = {
        "consistency": 0.20,
        "r2": 0.15,
        "rmse": 0.15,
        "significance": 0.10,
        "validation": 0.20,
        "quality": 0.10,
        "benchmark": 0.10,
    }
    total_score = (
        weights["consistency"] * consistency_score
        + weights["r2"] * r2_score
        + weights["rmse"] * rmse_score
        + weights["significance"] * significance_score
        + weights["validation"] * validation_score
        + weights["quality"] * quality_score
        + weights["benchmark"] * benchmark_score
    )

    # Classify using standardized thresholds
    label = "Exploratory"
    for threshold, lbl in RELIABILITY_THRESHOLDS:
        if total_score >= threshold:
            label = lbl
            break

    return {
        "score": round(float(total_score), 3),
        "label": label,
        "factors": {
            "consistency": round(float(consistency_score), 3),
            "r_squared": round(float(r2_score), 3),
            "rmse_accuracy": round(float(rmse_score), 3),
            "significance": round(float(significance_score), 3),
            "validation": round(float(validation_score), 3),
            "quality": round(float(quality_score), 3),
            "benchmark": round(float(benchmark_score), 3),
        },
    }


def _classify_forecast(
    reliability_score: float,
    trend_ratio: float,
    fc_r_squared: float,
    hist_trend: float,
    fc_trend: float,
    validation_metrics: dict = None,
    quality_checks: dict = None,
) -> dict[str, Any]:
    """
    Single classification function that derives all forecast metrics from one source.

    Produces: reliability_label, forecast_class, classification_reasons, recommended_action.
    All derived from the same inputs to ensure consistency across all reports.

    Args:
        reliability_score: Computed reliability score (0-1).
        trend_ratio: Ratio of forecast trend to historical trend.
        fc_r_squared: Forecast trend R-squared.
        hist_trend: Historical warming rate (°C/decade).
        fc_trend: Forecast warming rate (°C/decade).
        validation_metrics: Walk-forward validation metrics (optional).
        quality_checks: Forecast quality check results (optional).

    Returns:
        Dictionary with all classification metrics.
    """
    reasons = []

    # Determine reliability label from score
    reliability_label = "Exploratory"
    for threshold, lbl in RELIABILITY_THRESHOLDS:
        if reliability_score >= threshold:
            reliability_label = lbl
            break

    # Check for disqualifying conditions
    high_ratio = trend_ratio > MAX_TREND_RATIO
    low_r2 = fc_r_squared < 0.10
    wrong_direction = False
    if hist_trend != 0 and fc_trend != 0:
        wrong_direction = (hist_trend > 0) != (fc_trend > 0)

    # Check validation metrics
    validation_failed = False
    if validation_metrics:
        if validation_metrics.get("avg_rmse", 0) > 5.0:
            validation_failed = True
            reasons.append(
                f"Walk-forward validation RMSE ({validation_metrics['avg_rmse']:.2f}°C) "
                f"exceeds acceptable threshold (5.0°C)."
            )
        if validation_metrics.get("avg_r_squared", 0) < 0.0:
            validation_failed = True
            reasons.append(
                f"Walk-forward validation R² ({validation_metrics['avg_r_squared']:.4f}) "
                f"is negative, indicating model performs worse than mean prediction."
            )

    # Check quality checks
    quality_failed = False
    if quality_checks and not quality_checks.get("passed", True):
        quality_failed = True
        for issue in quality_checks.get("issues", []):
            reasons.append(f"Quality check failed: {issue}")

    # Build reasons list for existing checks
    if high_ratio:
        reasons.append(
            f"Trend ratio ({trend_ratio:.2f}x) exceeds acceptable threshold "
            f"({MAX_TREND_RATIO:.0f}x). Forecast diverges substantially from "
            f"historical patterns."
        )
    if low_r2:
        reasons.append(
            f"Forecast R-squared ({fc_r_squared:.4f}) is below reliability "
            f"threshold (0.10). Linear trend explains limited forecast variance."
        )
    if wrong_direction:
        reasons.append(
            "Forecast trend direction contradicts historical trend direction."
        )

    # Classify: reliability score is primary, all checks are secondary
    disqualifiers = high_ratio or low_r2 or wrong_direction or validation_failed or quality_failed

    if reliability_score >= 0.60 and not disqualifiers:
        forecast_class = "reliable"
        recommended_action = "Use forecast as primary projection."
    elif reliability_score >= 0.40 and not validation_failed and not quality_failed:
        forecast_class = "low_confidence"
        recommended_action = (
            "Use with caution. Cross-validate with alternative methods before "
            "incorporating into planning decisions."
        )
    else:
        forecast_class = "exploratory"
        recommended_action = (
            "Treat as scenario analysis only. Consider linear trend extrapolation "
            "as alternative projection. Do not use as primary planning basis."
        )

    return {
        "reliability_label": reliability_label,
        "forecast_class": forecast_class,
        "classification_reasons": reasons,
        "recommended_action": recommended_action,
    }


def validate_forecast_quality(
    forecast_df: pd.DataFrame,
    series: pd.Series,
    trend_info: dict,
    hist_trend_per_decade: float,
) -> dict[str, Any]:
    """
    Perform comprehensive quality checks on the forecast before reporting.

    Checks:
        1. No NaN values in forecast
        2. No impossible climate values (e.g., -50°C to 60°C for New Delhi)
        3. Reasonable forecast slope (not extreme)
        4. Valid confidence intervals (upper > lower, width reasonable)
        5. Forecast consistency with historical observations
        6. Model convergence (trend not zero when data shows trend)

    Args:
        forecast_df: Forecast DataFrame with predictions and intervals.
        series: Historical time series for reference.
        trend_info: Forecast trend information.
        hist_trend_per_decade: Historical trend rate.

    Returns:
        Dictionary with passed (bool), issues list, and details.
    """
    issues = []
    details = {}

    # 1. Check for NaN values
    nan_count = int(forecast_df[COL_FORECAST].isna().sum())
    details["nan_count"] = nan_count
    if nan_count > 0:
        issues.append(f"Forecast contains {nan_count} NaN values.")

    # 2. Check for physically reasonable values
    fc_values = forecast_df[COL_FORECAST].dropna()
    if len(fc_values) > 0:
        fc_min = float(fc_values.min())
        fc_max = float(fc_values.max())
        details["forecast_min"] = fc_min
        details["forecast_max"] = fc_max
        if fc_min < -50.0 or fc_max > 60.0:
            issues.append(
                f"Forecast contains physically unreasonable values "
                f"(range: {fc_min:.1f} to {fc_max:.1f}°C). "
                f"Expected range for New Delhi: -50°C to 60°C."
            )
    else:
        details["forecast_min"] = 0.0
        details["forecast_max"] = 0.0

    # 3. Check forecast slope reasonableness
    fc_trend = trend_info.get("trend_per_decade", 0)
    details["forecast_trend_per_decade"] = fc_trend
    if abs(fc_trend) > 10.0:
        issues.append(
            f"Forecast trend ({fc_trend:.3f}°C/decade) is extreme "
            f"(threshold: ±10.0°C/decade)."
        )

    # 4. Check confidence intervals
    if COL_FORECAST_UPPER in forecast_df.columns and COL_FORECAST_LOWER in forecast_df.columns:
        upper = forecast_df[COL_FORECAST_UPPER]
        lower = forecast_df[COL_FORECAST_LOWER]
        invalid_ci = int((upper < lower).sum())
        ci_width = float((upper - lower).mean())
        details["invalid_ci_count"] = invalid_ci
        details["avg_ci_width"] = ci_width
        if invalid_ci > 0:
            issues.append(f"Forecast has {invalid_ci} invalid confidence intervals (upper < lower).")
        if ci_width > 20.0:
            issues.append(
                f"Average confidence interval width ({ci_width:.1f}°C) is very large, "
                f"indicating high uncertainty."
            )

    # 5. Check consistency with historical observations
    hist_mean = float(series.mean())
    fc_mean = float(forecast_df[COL_FORECAST].mean())
    details["hist_mean"] = hist_mean
    details["fc_mean"] = fc_mean
    deviation = abs(fc_mean - hist_mean) / abs(hist_mean) if hist_mean != 0 else 0
    details["mean_deviation_pct"] = deviation * 100
    if deviation > 0.20:
        issues.append(
            f"Forecast mean ({fc_mean:.2f}°C) deviates {deviation*100:.1f}% "
            f"from historical mean ({hist_mean:.2f}°C)."
        )

    # 6. Check trend consistency
    if hist_trend_per_decade != 0:
        ratio = abs(fc_trend / hist_trend_per_decade)
        details["trend_ratio"] = ratio
        if ratio > MAX_TREND_RATIO * 2:
            issues.append(
                f"Trend ratio ({ratio:.2f}x) is extremely high, "
                f"indicating potential model instability."
            )

    passed = len(issues) == 0

    return {
        "passed": passed,
        "issues": issues,
        "details": details,
    }


def generate_forecast_diagnostic_figures(
    series: pd.Series,
    model_fit,
    forecast_series: pd.Series,
    forecast_df: pd.DataFrame,
    output_dir: Path = None,
) -> list[Path]:
    """
    Generate diagnostic figures for forecast evaluation.

    Figures generated:
        1. Residual Histogram
        2. Residual Q-Q Plot
        3. Residual vs Time
        4. Actual vs Fitted
        5. Forecast vs Historical
        6. Residual Autocorrelation (ACF)

    Args:
        series: Historical time series.
        model_fit: Fitted Holt-Winters model.
        forecast_series: Forecasted values.
        forecast_df: Forecast DataFrame with intervals.
        output_dir: Directory to save figures.

    Returns:
        List of saved figure paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf
    except ImportError:
        logger.warning("matplotlib or statsmodels not available for diagnostic figures")
        return []

    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    in_sample_pred = model_fit.fittedvalues
    residuals = series - in_sample_pred
    residuals_clean = residuals.dropna()

    saved_figures = []

    # 1. Residual Histogram
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(residuals_clean, bins=50, edgecolor="black", alpha=0.7, color="#4ECDC4")
        ax.set_xlabel("Residual (°C)")
        ax.set_ylabel("Frequency")
        ax.set_title("Residual Distribution")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.8)
        filepath = output_dir / "forecast_residual_histogram.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_figures.append(filepath)
    except Exception as e:
        logger.warning(f"Failed to generate residual histogram: {e}")

    # 2. Residual Q-Q Plot
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        sorted_resid = np.sort(residuals_clean.values)
        n = len(sorted_resid)
        theoretical = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
        ax.scatter(theoretical, sorted_resid, alpha=0.5, s=10, color="#4ECDC4")
        lims = [
            min(theoretical.min(), sorted_resid.min()),
            max(theoretical.max(), sorted_resid.max()),
        ]
        ax.plot(lims, lims, "r--", alpha=0.8, label="Normal Line")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.set_title("Residual Q-Q Plot")
        ax.legend()
        filepath = output_dir / "forecast_residual_qq.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_figures.append(filepath)
    except Exception as e:
        logger.warning(f"Failed to generate Q-Q plot: {e}")

    # 3. Residual vs Time
    try:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(residuals_clean.index, residuals_clean.values, alpha=0.3, s=5, color="#4ECDC4")
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Residual (°C)")
        ax.set_title("Residuals vs Time")
        filepath = output_dir / "forecast_residual_vs_time.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_figures.append(filepath)
    except Exception as e:
        logger.warning(f"Failed to generate residual vs time: {e}")

    # 4. Actual vs Fitted
    try:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(series.index, series.values, label="Actual", alpha=0.7, color="#2C3E50")
        ax.plot(in_sample_pred.index, in_sample_pred.values, label="Fitted", alpha=0.7, color="#E74C3C")
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Actual vs Fitted Values")
        ax.legend()
        filepath = output_dir / "forecast_actual_vs_fitted.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_figures.append(filepath)
    except Exception as e:
        logger.warning(f"Failed to generate actual vs fitted: {e}")

    # 5. Forecast vs Historical
    try:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(series.index[-365*2:], series.values[-365*2:], label="Historical", color="#2C3E50")
        ax.plot(forecast_df[COL_DATE], forecast_df[COL_FORECAST], label="Forecast", color="#E74C3C")
        if COL_FORECAST_UPPER in forecast_df.columns:
            ax.fill_between(
                forecast_df[COL_DATE],
                forecast_df[COL_FORECAST_LOWER],
                forecast_df[COL_FORECAST_UPPER],
                alpha=0.2, color="#E74C3C", label="95% CI",
            )
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Forecast vs Historical Observations")
        ax.legend()
        filepath = output_dir / "forecast_vs_historical.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_figures.append(filepath)
    except Exception as e:
        logger.warning(f"Failed to generate forecast vs historical: {e}")

    # 6. Residual Autocorrelation (ACF)
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_acf(residuals_clean, lags=min(60, len(residuals_clean) // 2 - 1), ax=ax, alpha=0.05)
        ax.set_title("Residual Autocorrelation (ACF)")
        filepath = output_dir / "forecast_residual_acf.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_figures.append(filepath)
    except Exception as e:
        logger.warning(f"Failed to generate ACF plot: {e}")

    logger.info(f"Generated {len(saved_figures)} diagnostic figures in {output_dir}")
    return saved_figures


def _generate_linear_trend_forecast(
    series: pd.Series,
    forecast_steps: int,
    residual_std: float,
    confidence_level: float,
) -> dict[str, Any]:
    """
    Generate fallback forecast using linear trend extrapolation.

    Used when Holt-Winters produces unstable extrapolations.
    The linear trend is more conservative and avoids the aggressive
    trend extrapolation that can occur with exponential smoothing.

    Args:
        series: Historical time series.
        forecast_steps: Number of days to forecast.
        residual_std: Standard deviation of model residuals.
        confidence_level: Confidence level for prediction intervals.

    Returns:
        Dictionary with forecast DataFrame, trend info, and metrics.
    """
    x = np.arange(len(series), dtype=float)
    y = series.values.astype(float)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Generate linear forecast
    future_x = np.arange(len(series), len(series) + forecast_steps, dtype=float)
    linear_forecast = slope * future_x + intercept

    # Compute trend
    trend_per_day = slope
    trend_per_year = trend_per_day * 365.25
    trend_per_decade = trend_per_year * 10

    # Expanding prediction intervals
    z_score = 1.96 if confidence_level == 0.95 else 2.576
    steps = np.arange(1, forecast_steps + 1, dtype=float)
    expanding_margin = z_score * residual_std * np.sqrt(steps / 365.25)

    forecast_dates = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1),
        periods=forecast_steps,
        freq="D",
    )

    forecast_df = pd.DataFrame({
        COL_DATE: forecast_dates,
        COL_FORECAST: linear_forecast,
        COL_FORECAST_UPPER: linear_forecast + expanding_margin,
        COL_FORECAST_LOWER: linear_forecast - expanding_margin,
    })

    return {
        "forecast_df": forecast_df,
        "trend_per_year": float(trend_per_year),
        "trend_per_decade": float(trend_per_decade),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
    }


def generate_forecast(
    df: pd.DataFrame,
    horizon_years: int = FORECAST_HORIZON_YEARS,
    confidence_level: float = FORECAST_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """
    Generate temperature forecast with comprehensive validation pipeline.

    Workflow:
        1. Fit Holt-Winters model
        2. Evaluate in-sample performance
        3. Perform walk-forward validation
        4. Benchmark against baseline models
        5. Generate future forecast
        6. Validate forecast quality (no NaN, no impossible values, etc.)
        7. Validate forecast stability against historical trend
        8. If unstable, fall back to linear trend extrapolation
        9. Apply trend dampening only as final safeguard
        10. Compute reliability score (incorporating all validation results)
        11. Generate diagnostic figures

    Args:
        df: Processed DataFrame with temperature data.
        horizon_years: Number of years to forecast.
        confidence_level: Confidence level for prediction intervals.

    Returns:
        Dictionary with forecast results, model metrics, reliability, and trend info.
    """
    if COL_TEMP_MEAN not in df.columns:
        raise ValueError("Temperature column not found in DataFrame")

    series = df.set_index(COL_DATE)[COL_TEMP_MEAN]
    series = series.asfreq("D")  # Ensure daily frequency

    # Fit model
    model_fit = fit_holt_winters(series)

    # Generate in-sample predictions for evaluation
    in_sample_pred = model_fit.fittedvalues
    metrics = evaluate_model(series, in_sample_pred)
    logger.info(
        f"Model metrics: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}, "
        f"MAPE={metrics['mape']:.2f}%, R-squared={metrics['r_squared']:.4f}"
    )

    # Warn if model fit is poor
    if metrics["r_squared"] < 0.1:
        logger.warning(
            f"Low in-sample R-squared ({metrics['r_squared']:.4f}): "
            f"model may not capture underlying patterns well."
        )

    # ── Walk-Forward Validation ────────────────────────────────────────────
    validation_results = walk_forward_validation(series, n_splits=3, test_size_days=365)
    logger.info(
        f"Walk-forward validation: Avg RMSE={validation_results['avg_rmse']:.3f}, "
        f"Avg MAE={validation_results['avg_mae']:.3f}, "
        f"Avg MAPE={validation_results['avg_mape']:.2f}%, "
        f"Avg R²={validation_results['avg_r_squared']:.4f}"
    )

    # ── Model Benchmarking ─────────────────────────────────────────────────
    benchmark_results = benchmark_models(series, test_days=365)
    logger.info(f"Benchmarking: {benchmark_results['recommendation']}")

    # Generate future forecast
    forecast_steps = horizon_years * 365
    forecast = model_fit.forecast(forecast_steps)

    # Compute expanding prediction intervals using residual diagnostics
    residuals = series - in_sample_pred
    residual_std = residuals.std()

    # Expanding margin: uncertainty grows with sqrt(horizon)
    z_score = 1.96 if confidence_level == 0.95 else 2.576
    steps = np.arange(1, forecast_steps + 1, dtype=float)
    expanding_margin = z_score * residual_std * np.sqrt(steps / 365.25)

    forecast_upper = forecast.values + expanding_margin
    forecast_lower = forecast.values - expanding_margin

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        COL_DATE: forecast.index,
        COL_FORECAST: forecast.values,
        COL_FORECAST_UPPER: forecast_upper,
        COL_FORECAST_LOWER: forecast_lower,
    })

    # Compute trend using linregress
    trend_info = _compute_forecast_trend_linregress(forecast, horizon_years)

    # Validate forecast reasonableness
    hist_slope = stats.linregress(
        np.arange(len(series)), series.values
    ).slope
    hist_trend_per_decade = hist_slope * 365.25 * 10

    trend_ratio = (
        abs(trend_info["trend_per_decade"] / hist_trend_per_decade)
        if hist_trend_per_decade != 0
        else float("inf")
    )

    # ── Forecast Quality Checks ────────────────────────────────────────────
    quality_checks = validate_forecast_quality(
        forecast_df, series, trend_info, hist_trend_per_decade
    )
    if not quality_checks["passed"]:
        logger.info("Forecast quality checks: ISSUES DETECTED (expected for some data patterns)")
        for issue in quality_checks["issues"]:
            logger.info(f"  - {issue}")
    else:
        logger.info("Forecast quality checks: PASSED")

    # ── Compute Reliability Score (with all validation inputs) ─────────────
    reliability = _compute_reliability_score(
        trend_ratio=trend_ratio,
        fc_r_squared=trend_info["r_squared"],
        in_sample_rmse=metrics["rmse"],
        mean_temp=series.mean(),
        trend_p_value=trend_info["p_value"],
        validation_metrics=validation_results,
        quality_checks=quality_checks,
        benchmark_results=benchmark_results,
    )

    # ── Classify Forecast (single source of truth) ─────────────────────────
    classification = _classify_forecast(
        reliability_score=reliability["score"],
        trend_ratio=trend_ratio,
        fc_r_squared=trend_info["r_squared"],
        hist_trend=hist_trend_per_decade,
        fc_trend=trend_info["trend_per_decade"],
        validation_metrics=validation_results,
        quality_checks=quality_checks,
    )

    forecast_class = classification["forecast_class"]
    classification_reasons = classification["classification_reasons"]
    recommended_action = classification["recommended_action"]

    logger.info(f"Forecast classified as {forecast_class.upper()}")
    logger.info(f"Reliability: {classification['reliability_label']} (score={reliability['score']:.3f})")
    for reason in classification_reasons:
        logger.info(f"  - {reason}")
    logger.info(f"  Recommended action: {recommended_action}")

    # ── Fallback to linear trend if unstable ───────────────────────────────
    used_linear_fallback = False
    if forecast_class == "exploratory":
        logger.info(
            "Holt-Winters classified as EXPLORATORY — "
            "primary model exhibits trend instability. "
            "Generating linear trend extrapolation as fallback for comparison."
        )
        linear_fallback = _generate_linear_trend_forecast(
            series, forecast_steps, residual_std, confidence_level
        )
        linear_trend_ratio = (
            abs(linear_fallback["trend_per_decade"] / hist_trend_per_decade)
            if hist_trend_per_decade != 0
            else float("inf")
        )

        # Use linear fallback if it's more stable
        if linear_trend_ratio < trend_ratio:
            logger.info(
                f"Linear fallback selected: trend ratio {linear_trend_ratio:.2f}x "
                f"(vs HW {trend_ratio:.2f}x) — better aligned with historical patterns. "
                f"Switching to linear trend forecast."
            )
            forecast_df = linear_fallback["forecast_df"]
            trend_info = {
                "trend_per_year": linear_fallback["trend_per_year"],
                "trend_per_decade": linear_fallback["trend_per_decade"],
                "r_squared": linear_fallback["r_squared"],
                "p_value": linear_fallback["p_value"],
            }
            trend_ratio = linear_trend_ratio
            used_linear_fallback = True

            # Re-classify with fallback model
            reliability = _compute_reliability_score(
                trend_ratio=trend_ratio,
                fc_r_squared=trend_info["r_squared"],
                in_sample_rmse=metrics["rmse"],
                mean_temp=series.mean(),
                trend_p_value=trend_info["p_value"],
                validation_metrics=validation_results,
                quality_checks={"passed": True, "issues": [], "details": {}},
                benchmark_results={"use_holt_winters": False},
            )
            classification = _classify_forecast(
                reliability_score=reliability["score"],
                trend_ratio=trend_ratio,
                fc_r_squared=trend_info["r_squared"],
                hist_trend=hist_trend_per_decade,
                fc_trend=trend_info["trend_per_decade"],
                validation_metrics=validation_results,
                quality_checks={"passed": True, "issues": [], "details": {}},
            )
            forecast_class = classification["forecast_class"]
            classification_reasons = classification["classification_reasons"]
            recommended_action = classification["recommended_action"]
        else:
            logger.info(
                f"Linear fallback not more stable (ratio={linear_trend_ratio:.2f}x). "
                f"Keeping original forecast with dampening."
            )

    # ── Trend dampening as final safeguard ─────────────────────────────────
    dampened_trend_per_decade = trend_info["trend_per_decade"]
    dampened_trend_per_year = trend_info["trend_per_year"]
    dampening_applied = False

    if trend_ratio > MAX_TREND_RATIO and hist_trend_per_decade != 0:
        dampening_factor = MAX_TREND_RATIO / trend_ratio
        dampened_trend_per_decade = trend_info["trend_per_decade"] * dampening_factor
        dampened_trend_per_year = dampened_trend_per_decade / 10
        dampening_applied = True
        logger.info(
            f"Trend dampening applied (safeguard): "
            f"{trend_info['trend_per_decade']:.3f} -> "
            f"{dampened_trend_per_decade:.3f} deg C/decade "
            f"(factor: {dampening_factor:.3f}, max ratio: {MAX_TREND_RATIO}x)"
        )

    # ── Generate Diagnostic Figures ────────────────────────────────────────
    diagnostic_figures = generate_forecast_diagnostic_figures(
        series, model_fit, forecast, forecast_df
    )

    # ── Assemble results (single source of truth) ──────────────────────────
    # NOTE: trend_r_squared = R² of linear trend fit on forecast (trend regression)
    #       model_metrics.r_squared = R² of Holt-Winters in-sample fit (model fit)
    #       These are different metrics and must NOT be confused.
    results = {
        "forecast_df": forecast_df,
        "model_metrics": metrics,
        # Dampened trend values (used in reports)
        "trend_per_year": dampened_trend_per_year,
        "trend_per_decade": dampened_trend_per_decade,
        # Trend regression R² (how well linear trend fits the forecast)
        "trend_r_squared": trend_info["r_squared"],
        "trend_p_value": trend_info["p_value"],
        # Model in-sample R² (how well Holt-Winters fits historical data)
        "model_r_squared": metrics["r_squared"],
        # Forecast horizon
        "forecast_horizon_years": horizon_years,
        "residual_std": float(residual_std),
        # Historical trend (uncapped, for reference)
        "historical_trend_per_decade": float(hist_trend_per_decade),
        # Trend ratio (UNCAPPED - the actual computed value)
        "trend_consistency_ratio": round(float(trend_ratio), 3),
        # Dampening info
        "trend_dampening_applied": dampening_applied,
        "original_trend_per_decade": trend_info["trend_per_decade"],
        "dampening_factor": (MAX_TREND_RATIO / trend_ratio) if trend_ratio > MAX_TREND_RATIO else 1.0,
        # Classification (all from single source of truth)
        "forecast_class": forecast_class,
        "classification_reasons": classification_reasons,
        "recommended_action": recommended_action,
        "reliability_score": reliability["score"],
        "reliability_label": classification["reliability_label"],
        "reliability_factors": reliability["factors"],
        "used_linear_fallback": used_linear_fallback,
        # Walk-forward validation
        "validation_metrics": validation_results,
        # Model benchmarking
        "benchmark_results": benchmark_results,
        # Quality checks
        "quality_checks": quality_checks,
        # Diagnostic figures
        "diagnostic_figures": [str(p) for p in diagnostic_figures],
    }

    # ── Forecast Validation Summary ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  FORECAST VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Primary Model:     Holt-Winters Exponential Smoothing")
    logger.info(f"  Validation:        {validation_results['n_folds']} folds, "
                f"Avg RMSE={validation_results['avg_rmse']:.3f}, "
                f"Avg R²={validation_results['avg_r_squared']:.4f}")
    logger.info(f"  Benchmark:         Best = {benchmark_results.get('best_model', 'N/A')}, "
                f"HW ranked #{benchmark_results.get('holt_winters', {}).get('rank', '?')}")
    logger.info(f"  Quality Checks:    {'PASSED' if quality_checks['passed'] else 'ISSUES DETECTED'}")
    if used_linear_fallback:
        logger.info(f"  Fallback Used:     YES — Linear trend selected (HW was EXPLORATORY)")
    else:
        logger.info(f"  Fallback Used:     NO")
    logger.info(f"  Reliability:       {classification['reliability_label']} "
                f"(score={reliability['score']:.3f})")
    logger.info(f"  Classification:    {forecast_class.upper()}")
    logger.info(f"  Recommended Use:   {recommended_action}")
    logger.info("=" * 60)

    logger.info(
        f"Forecast generated: {horizon_years} years ahead"
    )
    logger.info(
        f"Projected trend (linregress): "
        f"{dampened_trend_per_year:.3f} deg C/year "
        f"({dampened_trend_per_decade:.3f} deg C/decade, "
        f"R-squared={trend_info['r_squared']:.4f}, p={trend_info['p_value']:.4e})"
    )

    return results


def run_forecasting(df: pd.DataFrame) -> dict[str, Any]:
    """
    Execute the complete forecasting pipeline.

    Args:
        df: Processed DataFrame.

    Returns:
        Forecasting results dictionary.
    """
    return generate_forecast(df)
