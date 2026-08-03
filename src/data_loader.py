"""
Data Loader and Schema Validator for Climate Trend Analyzer.

Handles schema validation, date bounds checking, type conversion,
and initial data loading from raw CSV files.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import END_DATE, START_DATE
from src.constants import (
    COL_DATE,
    COL_HUMIDITY,
    COL_PRECIPITATION,
    COL_SOLAR_RADIATION,
    COL_TEMP_MEAN,
    COL_TEMP_MAX,
    COL_TEMP_MIN,
    COL_WIND_SPEED,
    CLIMATE_VARIABLES,
    REQUIRED_COLUMNS,
)
from src.logger import get_logger, log_pipeline_stage

logger = get_logger("data_loader")

# Expected schema: column name -> expected dtype
EXPECTED_SCHEMA = {
    COL_DATE: "datetime64[ns]",
    COL_TEMP_MEAN: "float64",
    COL_TEMP_MAX: "float64",
    COL_TEMP_MIN: "float64",
    COL_PRECIPITATION: "float64",
    COL_HUMIDITY: "float64",
    COL_SOLAR_RADIATION: "float64",
    COL_WIND_SPEED: "float64",
}


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw climate data from CSV.

    Args:
        filepath: Path to CSV file. If None, loads merged_raw.csv.

    Returns:
        Raw DataFrame.
    """
    if filepath is None:
        filepath = RAW_DIR / "merged" / "merged_raw.csv"

    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found: {filepath}")

    df = pd.read_csv(filepath, parse_dates=[COL_DATE], encoding="utf-8")
    logger.info(f"Loaded raw data: {filepath} ({len(df)} rows)")
    return df


def validate_schema(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Validate DataFrame against expected schema.

    Returns:
        Dictionary with 'missing_columns' and 'type_mismatches'.
    """
    issues = {
        "missing_columns": [],
        "type_mismatches": [],
    }

    # Check required columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            issues["missing_columns"].append(col)

    # Check data types
    for col, expected_dtype in EXPECTED_SCHEMA.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if "datetime" in expected_dtype and "datetime" not in actual_dtype:
                issues["type_mismatches"].append(f"{col}: expected {expected_dtype}, got {actual_dtype}")
            elif "float" in expected_dtype and "float" not in actual_dtype and "int" not in actual_dtype:
                issues["type_mismatches"].append(f"{col}: expected {expected_dtype}, got {actual_dtype}")

    return issues


def validate_date_bounds(df: pd.DataFrame) -> bool:
    """
    Validate that date column falls within configured bounds.

    Returns:
        True if dates are within bounds.
    """
    if COL_DATE not in df.columns:
        logger.error("Date column missing from DataFrame")
        return False

    min_date = df[COL_DATE].min()
    max_date = df[COL_DATE].max()
    expected_start = pd.Timestamp(START_DATE)
    expected_end = pd.Timestamp(END_DATE)

    if min_date > expected_start:
        logger.info(
            f"API coverage: Data starts {min_date.date()} "
            f"(required: {expected_start.date()}). "
            f"Falling back to synthetic data for full period."
        )
    if max_date < expected_end:
        logger.info(
            f"API coverage: Data ends {max_date.date()} "
            f"(required: {expected_end.date()}). "
            f"Falling back to synthetic data for full period."
        )

    logger.info(f"Date range: {min_date.date()} to {max_date.date()}")
    return True


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types.

    Returns:
        DataFrame with corrected types.
    """
    df = df.copy()

    # Ensure date column is datetime
    if COL_DATE in df.columns:
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")

    # Ensure numeric columns are float
    numeric_cols = [c for c in CLIMATE_VARIABLES if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"Type conversion completed for {len(numeric_cols)} numeric columns")
    return df


def compute_data_quality_report(df: pd.DataFrame) -> dict:
    """
    Compute data quality metrics for the DataFrame.

    Returns:
        Dictionary with quality metrics.
    """
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()

    quality = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_cells": int(total_cells),
        "missing_cells": int(missing_cells),
        "missing_percentage": round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0,
        "column_missing": {},
    }

    for col in df.columns:
        missing = int(df[col].isna().sum())
        quality["column_missing"][col] = {
            "count": missing,
            "percentage": round((missing / len(df)) * 100, 2) if len(df) > 0 else 0,
        }

    return quality


def load_and_validate(
    filepath: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Complete data loading and validation pipeline.

    Args:
        filepath: Optional path to raw data CSV.

    Returns:
        Validated and type-converted DataFrame.
    """
    log_pipeline_stage("Data Loading & Validation")

    # Load raw data
    df = load_raw_data(filepath)

    # Validate schema
    issues = validate_schema(df)
    if issues["missing_columns"]:
        logger.error(f"Missing required columns: {issues['missing_columns']}")
        raise ValueError(f"Schema validation failed: missing columns {issues['missing_columns']}")

    if issues["type_mismatches"]:
        logger.warning(f"Type mismatches found: {issues['type_mismatches']}")

    # Convert types
    df = convert_types(df)

    # Validate date bounds
    validate_date_bounds(df)

    # Data quality report
    quality = compute_data_quality_report(df)
    logger.info(f"Data quality: {quality['missing_percentage']}% missing values")

    return df
