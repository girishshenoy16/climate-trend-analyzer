"""
Reusable Utility Helpers for Climate Trend Analyzer.

Provides safe JSON/CSV I/O, directory creation, date formatting,
and other common helper functions used across the pipeline.
"""

import json
from pathlib import Path
from datetime import datetime, date
from typing import Any, Optional, Union

import pandas as pd

from src.logger import get_logger

logger = get_logger("utils")


def ensure_directory(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Safely write data to a JSON file.

    Args:
        data: Serializable Python object.
        filepath: Target file path.
        indent: JSON indentation level.
    """
    filepath = Path(filepath)
    ensure_directory(filepath.parent)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str, ensure_ascii=False)

    logger.debug(f"JSON saved: {filepath}")


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Safely read and return data from a JSON file.

    Args:
        filepath: Source file path.

    Returns:
        Parsed JSON data.

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If file contains invalid JSON.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.debug(f"JSON loaded: {filepath}")
    return data


def save_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    index: bool = False,
) -> None:
    """
    Safely write a DataFrame to CSV.

    Args:
        df: Pandas DataFrame to save.
        filepath: Target file path.
        index: Whether to write row indices.
    """
    filepath = Path(filepath)
    ensure_directory(filepath.parent)

    df.to_csv(filepath, index=index, encoding="utf-8")
    logger.debug(f"CSV saved: {filepath} ({len(df)} rows)")


def load_csv(
    filepath: Union[str, Path],
    parse_dates: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Safely read a CSV file into a DataFrame.

    Args:
        filepath: Source file path.
        parse_dates: Columns to parse as dates.

    Returns:
        Pandas DataFrame.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    df = pd.read_csv(filepath, parse_dates=parse_dates, encoding="utf-8")
    logger.debug(f"CSV loaded: {filepath} ({len(df)} rows, {len(df.columns)} cols)")
    return df


def format_date(dt: Union[datetime, date, str], fmt: str = "%Y-%m-%d") -> str:
    """
    Format a date/datetime object to string.

    Args:
        dt: Date or datetime object, or date string.
        fmt: Output format string.

    Returns:
        Formatted date string.
    """
    if isinstance(dt, str):
        dt = datetime.strptime(dt, "%Y-%m-%d")
    elif isinstance(dt, date) and not isinstance(dt, datetime):
        dt = datetime(dt.year, dt.month, dt.day)
    return dt.strftime(fmt)


def parse_date(date_str: str, fmt: str = "%Y-%m-%d") -> datetime:
    """Parse a date string into a datetime object."""
    return datetime.strptime(date_str, fmt)


def compute_summary_stats(series: pd.Series) -> dict[str, float]:
    """Compute basic summary statistics for a numeric Series."""
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
        "q25": float(series.quantile(0.25)),
        "q75": float(series.quantile(0.75)),
        "count": int(series.count()),
    }


def normalize_column_name(name: str) -> str:
    """Normalize column name to snake_case."""
    return (
        name.lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator
