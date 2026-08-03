"""
Hybrid Climate Data Acquisition Engine for Climate Trend Analyzer.

Connects to NASA POWER API and Open-Meteo Climate Archive API,
handles retries and HTTP caching, converts JSON payloads to
structured DataFrames, and merges datasets into a unified format.
"""

import hashlib
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.config import (
    CACHE_DIR,
    CACHE_ENABLED,
    CACHE_EXPIRY_HOURS,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    END_DATE,
    MAX_RETRIES,
    NASA_POWER_BASE_URL,
    NASA_POWER_PARAMETERS,
    OPEN_METEO_BASE_URL,
    OPEN_METEO_DAILY_VARS,
    RAW_DIR,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF_FACTOR,
    START_DATE,
)
from src.constants import (
    NASA_POWER_COLUMN_MAP,
    OPEN_METEO_COLUMN_MAP,
)
from src.logger import get_logger, log_pipeline_stage

logger = get_logger("api_fetcher")


class HTTPCache:
    """Local file-based HTTP response cache."""

    def __init__(self, cache_dir: Path = CACHE_DIR, expiry_hours: int = CACHE_EXPIRY_HOURS):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expiry_hours = expiry_hours

    def _key(self, url: str, params: Optional[dict] = None) -> str:
        """Generate a unique cache key from URL and parameters."""
        content = url + str(sorted(params.items()) if params else "")
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """Retrieve cached response if valid."""
        if not CACHE_ENABLED:
            return None

        key = self._key(url, params)
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        # Check expiry
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age > self.expiry_hours * 3600:
            cache_file.unlink(missing_ok=True)
            return None

        try:
            import json
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def put(self, url: str, data: dict, params: Optional[dict] = None) -> None:
        """Store response in cache."""
        if not CACHE_ENABLED:
            return

        key = self._key(url, params)
        cache_file = self.cache_dir / f"{key}.json"

        import json
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str, ensure_ascii=False)


class NASAPowerFetcher:
    """NASA POWER API client for daily climate data."""

    def __init__(self, lat: float = DEFAULT_LATITUDE, lon: float = DEFAULT_LONGITUDE):
        self.lat = lat
        self.lon = lon
        self.cache = HTTPCache()

    def fetch(
        self,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
    ) -> pd.DataFrame:
        """
        Fetch daily climate data from NASA POWER API.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            DataFrame with normalized column names.
        """
        log_pipeline_stage("NASA POWER API Fetch")

        params = {
            "parameters": ",".join(NASA_POWER_PARAMETERS),
            "community": "RE",
            "longitude": self.lon,
            "latitude": self.lat,
            "start": start_date.replace("-", ""),
            "end": end_date.replace("-", ""),
            "format": "JSON",
        }

        # Check cache
        cached = self.cache.get(NASA_POWER_BASE_URL, params)
        if cached:
            logger.info("Using cached NASA POWER data")
            return self._parse_response(cached)

        # Fetch with retries
        response = self._fetch_with_retry(NASA_POWER_BASE_URL, params)
        self.cache.put(NASA_POWER_BASE_URL, response, params)

        return self._parse_response(response)

    def _fetch_with_retry(self, url: str, params: dict) -> dict:
        """Execute HTTP request with exponential backoff retry."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"NASA POWER request attempt {attempt}/{MAX_RETRIES}")
                resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                wait = RETRY_BACKOFF_FACTOR * (2 ** (attempt - 1))
                logger.warning(f"Request failed: {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)

        raise ConnectionError(f"NASA POWER API failed after {MAX_RETRIES} retries")

    def _parse_response(self, data: dict) -> pd.DataFrame:
        """Parse NASA POWER JSON response into DataFrame."""
        properties = data.get("properties", {})
        parameter_data = properties.get("parameter", {})

        if not parameter_data:
            raise ValueError("No parameter data found in NASA POWER response")

        # Extract dates from the first parameter
        first_param = list(parameter_data.values())[0]
        dates = sorted(first_param.keys())

        records = []
        for date_str in dates:
            record = {"date": pd.to_datetime(date_str, format="%Y%m%d")}
            for api_col, df_col in NASA_POWER_COLUMN_MAP.items():
                value = parameter_data.get(api_col, {}).get(date_str)
                record[df_col] = value if value is not None else None
            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"NASA POWER: Parsed {len(df)} daily records")

        # Save raw JSON
        raw_dir = RAW_DIR / "nasa_power"
        raw_dir.mkdir(parents=True, exist_ok=True)
        from src.utils import save_json, save_csv
        save_json(data, raw_dir / "nasa_power_raw.json")
        save_csv(df, raw_dir / "nasa_power_raw.csv")

        return df


class OpenMeteoFetcher:
    """Open-Meteo Climate Archive API client."""

    def __init__(self, lat: float = DEFAULT_LATITUDE, lon: float = DEFAULT_LONGITUDE):
        self.lat = lat
        self.lon = lon
        self.cache = HTTPCache()

    def fetch(
        self,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
    ) -> pd.DataFrame:
        """
        Fetch daily climate data from Open-Meteo Archive API.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            DataFrame with normalized column names.
        """
        log_pipeline_stage("Open-Meteo Climate Archive Fetch")

        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(OPEN_METEO_DAILY_VARS),
            "timezone": "Asia/Kolkata",
        }

        # Check cache
        cached = self.cache.get(OPEN_METEO_BASE_URL, params)
        if cached:
            logger.info("Using cached Open-Meteo data")
            return self._parse_response(cached)

        # Fetch with retries
        response = self._fetch_with_retry(OPEN_METEO_BASE_URL, params)
        self.cache.put(OPEN_METEO_BASE_URL, response, params)

        return self._parse_response(response)

    def _fetch_with_retry(self, url: str, params: dict) -> dict:
        """Execute HTTP request with exponential backoff retry."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Open-Meteo request attempt {attempt}/{MAX_RETRIES}")
                resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                wait = RETRY_BACKOFF_FACTOR * (2 ** (attempt - 1))
                logger.warning(f"Request failed: {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)

        raise ConnectionError(f"Open-Meteo API failed after {MAX_RETRIES} retries")

    def _parse_response(self, data: dict) -> pd.DataFrame:
        """Parse Open-Meteo JSON response into DataFrame."""
        daily = data.get("daily", {})

        if not daily:
            raise ValueError("No daily data found in Open-Meteo response")

        dates = pd.to_datetime(daily["time"])
        records = []

        for i, date_val in enumerate(dates):
            record = {"date": date_val}
            for api_col, df_col in OPEN_METEO_COLUMN_MAP.items():
                values = daily.get(api_col, [])
                record[df_col] = values[i] if i < len(values) else None
            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Open-Meteo: Parsed {len(df)} daily records")

        # Save raw JSON and CSV
        raw_dir = RAW_DIR / "open_meteo"
        raw_dir.mkdir(parents=True, exist_ok=True)
        from src.utils import save_json, save_csv
        save_json(data, raw_dir / "open_meteo_raw.json")
        save_csv(df, raw_dir / "open_meteo_raw.csv")

        return df


def merge_datasets(
    nasa_df: pd.DataFrame,
    openmeteo_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge NASA POWER and Open-Meteo DataFrames into a unified dataset.
    Uses Open-Meteo as primary source with NASA POWER as fallback for missing values.

    Args:
        nasa_df: NASA POWER DataFrame.
        openmeteo_df: Open-Meteo DataFrame.

    Returns:
        Merged and cleaned DataFrame.
    """
    log_pipeline_stage("Dataset Merging")

    # Normalize date columns
    nasa_df["date"] = pd.to_datetime(nasa_df["date"])
    openmeteo_df["date"] = pd.to_datetime(openmeteo_df["date"])

    # Merge on date
    merged = pd.merge(
        openmeteo_df,
        nasa_df,
        on="date",
        how="outer",
        suffixes=("_openmeteo", "_nasa"),
    )

    # For each climate variable, prefer Open-Meteo, fallback to NASA POWER
    base_cols = ["temperature", "temperature_max", "temperature_min",
                 "precipitation", "humidity", "solar_radiation", "wind_speed"]

    for col in base_cols:
        om_col = f"{col}_openmeteo"
        nasa_col = f"{col}_nasa"

        if om_col in merged.columns and nasa_col in merged.columns:
            merged[col] = merged[om_col].fillna(merged[nasa_col])
            merged.drop(columns=[om_col, nasa_col], inplace=True)
        elif om_col in merged.columns:
            merged.rename(columns={om_col: col}, inplace=True)
        elif nasa_col in merged.columns:
            merged.rename(columns={nasa_col: col}, inplace=True)

    # Sort by date
    merged.sort_values("date", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Save merged dataset
    merged_dir = RAW_DIR / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    from src.utils import save_csv
    save_csv(merged, merged_dir / "merged_raw.csv")

    logger.info(f"Merged dataset: {len(merged)} rows, {len(merged.columns)} columns")
    return merged


def fetch_all_data(
    lat: float = DEFAULT_LATITUDE,
    lon: float = DEFAULT_LONGITUDE,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """
    Execute the full data acquisition pipeline.

    Fetches from both APIs, merges results, and saves raw data.

    Args:
        lat: Latitude coordinate.
        lon: Longitude coordinate.
        start_date: Start date string.
        end_date: End date string.

    Returns:
        Merged DataFrame with all climate variables.
    """
    # Fetch from NASA POWER
    nasa_fetcher = NASAPowerFetcher(lat=lat, lon=lon)
    nasa_df = nasa_fetcher.fetch(start_date=start_date, end_date=end_date)

    # Fetch from Open-Meteo
    openmeteo_fetcher = OpenMeteoFetcher(lat=lat, lon=lon)
    openmeteo_df = openmeteo_fetcher.fetch(start_date=start_date, end_date=end_date)

    # Merge datasets
    merged_df = merge_datasets(nasa_df, openmeteo_df)

    return merged_df
