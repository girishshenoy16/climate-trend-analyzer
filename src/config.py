"""
Central Project Configuration Module for Climate Trend Analyzer.

Provides all configurable parameters, file paths, API endpoints,
default coordinates, and date range settings used across the pipeline.
"""

from pathlib import Path

# ─── Base Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"

# ─── Pipeline Version ────────────────────────────────────────────────────────
PIPELINE_VERSION = "2.0.0"

# ─── Default Geographic Coordinates ──────────────────────────────────────────
DEFAULT_LATITUDE = 28.6139   # New Delhi, India
DEFAULT_LONGITUDE = 77.2090
DEFAULT_ELEVATION = 216      # meters

# ─── Date Range ──────────────────────────────────────────────────────────────
START_YEAR = 2015
END_YEAR = 2024
START_DATE = f"{START_YEAR}-01-01"
END_DATE = f"{END_YEAR}-12-31"

# ─── NASA POWER API Configuration ────────────────────────────────────────────
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_PARAMETERS = [
    "T2M",        # Temperature at 2 Meters (°C)
    "T2M_MAX",    # Temperature at 2 Meters Maximum (°C)
    "T2M_MIN",    # Temperature at 2 Meters Minimum (°C)
    "PRECTOTCORR",# Precipitation Corrected (mm/day)
    "RH2M",       # Relative Humidity at 2 Meters (%)
    "ALLSKY_SFC_SW_DWN",  # Solar Radiation (MJ/m²/day)
    "WS2M",       # Wind Speed at 2 Meters (m/s)
]

# ─── Open-Meteo API Configuration ────────────────────────────────────────────
OPEN_METEO_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "shortwave_radiation_sum",
    "windspeed_10m_max",
]

# ─── HTTP Cache Settings ─────────────────────────────────────────────────────
CACHE_ENABLED = True
CACHE_EXPIRY_HOURS = 24
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 0.5
REQUEST_TIMEOUT = 30

# ─── Processing Settings ─────────────────────────────────────────────────────
ROLLING_WINDOWS = [7, 30, 365]
MISSING_VALUE_METHOD = "linear"    # interpolation method
SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

# ─── Forecasting Settings ────────────────────────────────────────────────────
FORECAST_HORIZON_YEARS = 3
FORECAST_CONFIDENCE_LEVEL = 0.95
HOLT_WINTERS_SEASONAL_PERIODS = 365
MAX_TREND_RATIO = 3.0  # Maximum acceptable ratio between forecast and historical trends

# ─── Anomaly Detection Settings ─────────────────────────────────────────────
ZSCORE_THRESHOLD = 2.5
ISOLATION_FOREST_CONTAMINATION = 0.05
ISOLATION_FOREST_RANDOM_STATE = 42

# ─── Visualization Settings ─────────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
FIGURE_BBOX_INCHES = "tight"

# ─── Simulation Mode ────────────────────────────────────────────────────────
SIMULATION_ENABLED = False
SYNTHETIC_START_YEAR = 2015
SYNTHETIC_END_YEAR = 2024

# ─── Climate Risk Score Weights ─────────────────────────────────────────────
RISK_WEIGHT_TEMP_TREND = 0.40
RISK_WEIGHT_RAINFALL_DEV = 0.30
RISK_WEIGHT_ANOMALY_FREQ = 0.30
RISK_CATEGORIES = {
    "Low": (0.00, 0.25),
    "Moderate": (0.25, 0.50),
    "High": (0.50, 0.75),
    "Very High": (0.75, 1.00),
}

# ─── Station Metadata ───────────────────────────────────────────────────────
STATION_NAME = "New Delhi, India"
STATION_ID = "IND_001"
STATION_LAT = DEFAULT_LATITUDE
STATION_LON = DEFAULT_LONGITUDE


def get_data_directories() -> dict[str, Path]:
    """Return all data-related directory paths."""
    return {
        "data": DATA_DIR,
        "raw": RAW_DIR,
        "processed": PROCESSED_DIR,
        "cache": CACHE_DIR,
        "docs_data": DOCS_DATA_DIR,
        "figures": FIGURES_DIR,
        "reports": REPORTS_DIR,
    }


def ensure_directories() -> None:
    """Create all required directories if they don't exist."""
    dirs = get_data_directories()
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
