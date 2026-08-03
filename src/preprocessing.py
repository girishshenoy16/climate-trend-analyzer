"""
Preprocessing & Feature Engineering for Climate Trend Analyzer.

Handles missing value imputation, temporal feature creation,
rolling moving averages, lag features, and dataset export.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.config import (
    MISSING_VALUE_METHOD,
    PROCESSED_DIR,
    ROLLING_WINDOWS,
    SEASON_MAP,
)
from src.constants import (
    COL_DATE,
    COL_DAY,
    COL_MONTH,
    COL_PRECIPITATION,
    COL_SEASON,
    COL_TEMP_LAG1,
    COL_TEMP_LAG30,
    COL_TEMP_LAG7,
    COL_TEMP_MEAN,
    COL_YEAR,
    SEASONS,
)
from src.logger import get_logger, log_pipeline_stage

logger = get_logger("preprocessing")


def impute_missing_values(
    df: pd.DataFrame,
    method: str = MISSING_VALUE_METHOD,
) -> pd.DataFrame:
    """
    Impute missing values using specified interpolation method.

    Args:
        df: Input DataFrame.
        method: Interpolation method ('linear', 'time', 'spline').

    Returns:
        DataFrame with imputed values.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    missing_before = df[numeric_cols].isna().sum().sum()

    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].interpolate(method=method, limit_direction="both")

    # Forward/backward fill any remaining NaN
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    missing_after = df[numeric_cols].isna().sum().sum()
    logger.info(f"Imputation: {missing_before} -> {missing_after} missing values (method={method})")

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal feature columns (Year, Month, Day of Year, Season).

    Args:
        df: Input DataFrame with 'date' column.

    Returns:
        DataFrame with temporal features added.
    """
    df = df.copy()

    df[COL_YEAR] = df[COL_DATE].dt.year
    df[COL_MONTH] = df[COL_DATE].dt.month
    df[COL_DAY] = df[COL_DATE].dt.dayofyear
    df[COL_SEASON] = df[COL_MONTH].map(SEASON_MAP)

    logger.info(f"Temporal features added: Year, Month, Day of Year, Season")
    return df


def add_rolling_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling moving averages for temperature and precipitation.

    Computes 7-day, 30-day, and 365-day rolling means.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with rolling average columns.
    """
    df = df.copy()

    if COL_TEMP_MEAN in df.columns:
        for window in ROLLING_WINDOWS:
            col_name = f"temperature_ma{window}"
            df[col_name] = df[COL_TEMP_MEAN].rolling(
                window=window, min_periods=1, center=False
            ).mean()
        logger.info("Temperature rolling averages computed (7, 30, 365 days)")

    if COL_PRECIPITATION in df.columns:
        for window in [7, 30]:
            col_name = f"precipitation_ma{window}"
            df[col_name] = df[COL_PRECIPITATION].rolling(
                window=window, min_periods=1, center=False
            ).mean()
        logger.info("Precipitation rolling averages computed (7, 30 days)")

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features for temperature time-series.

    Creates lag-1, lag-7, and lag-30 day features.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with lag feature columns.
    """
    df = df.copy()

    if COL_TEMP_MEAN in df.columns:
        df[COL_TEMP_LAG1] = df[COL_TEMP_MEAN].shift(1)
        df[COL_TEMP_LAG7] = df[COL_TEMP_MEAN].shift(7)
        df[COL_TEMP_LAG30] = df[COL_TEMP_MEAN].shift(30)

        # Fill initial NaN from lagging
        df[COL_TEMP_LAG1] = df[COL_TEMP_LAG1].bfill()
        df[COL_TEMP_LAG7] = df[COL_TEMP_LAG7].bfill()
        df[COL_TEMP_LAG30] = df[COL_TEMP_LAG30].bfill()

        logger.info("Lag features added (1, 7, 30 days)")

    return df


def export_processed_dataset(
    df: pd.DataFrame,
    filepath: Optional[Path] = None,
) -> Path:
    """
    Export the processed DataFrame to CSV.

    Args:
        df: Processed DataFrame.
        filepath: Output file path. Defaults to processed/climate_daily_processed.csv.

    Returns:
        Path to exported file.
    """
    if filepath is None:
        filepath = PROCESSED_DIR / "climate_daily_processed.csv"

    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False, encoding="utf-8")
    logger.info(f"Processed dataset exported: {filepath} ({len(df)} rows, {len(df.columns)} columns)")

    return filepath


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the complete preprocessing pipeline.

    Steps:
        1. Impute missing values
        2. Add temporal features
        3. Add rolling averages
        4. Add lag features

    Args:
        df: Raw input DataFrame.

    Returns:
        Fully processed DataFrame.
    """
    log_pipeline_stage("Data Preprocessing")

    df = impute_missing_values(df)
    df = add_temporal_features(df)
    df = add_rolling_averages(df)
    df = add_lag_features(df)

    logger.info(f"Preprocessing complete: {len(df)} rows, {len(df.columns)} columns")
    return df
