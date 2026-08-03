"""
Constants Module for Climate Trend Analyzer.

Defines column names, measurement units, color palettes,
anomaly thresholds, and other project-wide constants.
"""

# ─── Column Names (Normalized) ───────────────────────────────────────────────
COL_DATE = "date"
COL_YEAR = "year"
COL_MONTH = "month"
COL_DAY = "day_of_year"
COL_SEASON = "season"

# Temperature
COL_TEMP_MEAN = "temperature"
COL_TEMP_MAX = "temperature_max"
COL_TEMP_MIN = "temperature_min"

# Precipitation
COL_PRECIPITATION = "precipitation"

# Humidity
COL_HUMIDITY = "humidity"

# Solar Radiation
COL_SOLAR_RADIATION = "solar_radiation"

# Wind
COL_WIND_SPEED = "wind_speed"

# Rolling averages
COL_TEMP_MA7 = "temperature_ma7"
COL_TEMP_MA30 = "temperature_ma30"
COL_TEMP_MA365 = "temperature_ma365"
COL_PRECIP_MA7 = "precipitation_ma7"
COL_PRECIP_MA30 = "precipitation_ma30"

# Lag features
COL_TEMP_LAG1 = "temperature_lag1"
COL_TEMP_LAG7 = "temperature_lag7"
COL_TEMP_LAG30 = "temperature_lag30"

# Anomaly flags
COL_ANOMALY_ZSCORE = "anomaly_zscore"
COL_ANOMALY_IFOREST = "anomaly_iforest"
COL_ANOMALY_COMBINED = "anomaly_combined"

# Trend components
COL_TREND = "trend"
COL_SEASONAL = "seasonal_component"
COL_RESIDUAL = "residual"

# Forecast columns
COL_FORECAST = "forecast"
COL_FORECAST_LOWER = "forecast_lower"
COL_FORECAST_UPPER = "forecast_upper"

# All primary climate variable columns
CLIMATE_VARIABLES = [
    COL_TEMP_MEAN,
    COL_TEMP_MAX,
    COL_TEMP_MIN,
    COL_PRECIPITATION,
    COL_HUMIDITY,
    COL_SOLAR_RADIATION,
    COL_WIND_SPEED,
]

# Required columns for processing pipeline
REQUIRED_COLUMNS = [
    COL_DATE,
    COL_TEMP_MEAN,
    COL_PRECIPITATION,
]

# ─── Measurement Units ────────────────────────────────────────────────────────
UNITS = {
    COL_TEMP_MEAN: "°C",
    COL_TEMP_MAX: "°C",
    COL_TEMP_MIN: "°C",
    COL_PRECIPITATION: "mm/day",
    COL_HUMIDITY: "%",
    COL_SOLAR_RADIATION: "MJ/m²/day",
    COL_WIND_SPEED: "m/s",
}

# ─── NASA POWER API Column Mapping ────────────────────────────────────────────
NASA_POWER_COLUMN_MAP = {
    "T2M": COL_TEMP_MEAN,
    "T2M_MAX": COL_TEMP_MAX,
    "T2M_MIN": COL_TEMP_MIN,
    "PRECTOTCORR": COL_PRECIPITATION,
    "RH2M": COL_HUMIDITY,
    "ALLSKY_SFC_SW_DWN": COL_SOLAR_RADIATION,
    "WS2M": COL_WIND_SPEED,
}

# ─── Open-Meteo API Column Mapping ────────────────────────────────────────────
OPEN_METEO_COLUMN_MAP = {
    "temperature_2m_mean": COL_TEMP_MEAN,
    "temperature_2m_max": COL_TEMP_MAX,
    "temperature_2m_min": COL_TEMP_MIN,
    "precipitation_sum": COL_PRECIPITATION,
    "relative_humidity_2m_mean": COL_HUMIDITY,
    "shortwave_radiation_sum": COL_SOLAR_RADIATION,
    "windspeed_10m_max": COL_WIND_SPEED,
}

# ─── Color Palettes ───────────────────────────────────────────────────────────
COLORS = {
    "primary": "#1E3A5F",
    "secondary": "#4A90D9",
    "accent": "#F39C12",
    "danger": "#E74C3C",
    "success": "#27AE60",
    "warning": "#F1C40F",
    "background": "#0F1B2D",
    "card": "#1A2940",
    "text": "#FFFFFF",
    "text_muted": "#A0B4C8",
    "grid": "#1A1A2E",
}

TEMPERATURE_PALETTE = {
    "mean": "#E74C3C",
    "max": "#FF6B6B",
    "min": "#74B9FF",
    "fill": "rgba(231, 76, 60, 0.2)",
}

CHART_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#34495E",
]

# ─── Season Definitions ───────────────────────────────────────────────────────
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
SEASON_COLORS = {
    "Winter": "#74B9FF",
    "Spring": "#2ECC71",
    "Summer": "#E74C3C",
    "Autumn": "#F39C12",
}

# ─── Anomaly Thresholds ───────────────────────────────────────────────────────
ZSCORE_THRESHOLD = 2.5
IQR_MULTIPLIER = 1.5

# ─── Risk Score Categories ───────────────────────────────────────────────────
RISK_CATEGORIES = {
    "Low": (0.00, 0.25),
    "Moderate": (0.25, 0.50),
    "High": (0.50, 0.75),
    "Very High": (0.75, 1.00),
}

RISK_COLORS = {
    "Low": "#27AE60",
    "Moderate": "#F1C40F",
    "High": "#E67E22",
    "Very High": "#E74C3C",
}

# ─── Report Metadata ─────────────────────────────────────────────────────────
REPORT_TITLE = "Climate Trend Analyzer - Executive Report"
REPORT_SUBTITLE = "Automated Climate Analysis & Forecasting System"
REPORT_ORGANIZATION = "Climate Analytics Division"
REPORT_AUTHOR = "Climate Trend Analyzer Pipeline"
