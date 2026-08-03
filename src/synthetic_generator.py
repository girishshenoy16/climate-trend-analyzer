"""
Synthetic Climate Scenario Generator for Climate Trend Analyzer.

Generates realistic multi-year synthetic climate data for testing,
simulation, and fallback when API endpoints are unavailable.
Uses sinusoidal seasonal patterns with realistic noise and warming trends.
"""

import numpy as np
import pandas as pd
from datetime import datetime

from src.config import (
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    SYNTHETIC_END_YEAR,
    SYNTHETIC_START_YEAR,
)
from src.constants import (
    COL_DATE,
    COL_HUMIDITY,
    COL_PRECIPITATION,
    COL_SOLAR_RADIATION,
    COL_TEMP_MEAN,
    COL_TEMP_MAX,
    COL_TEMP_MIN,
    COL_WIND_SPEED,
)
from src.logger import get_logger, log_pipeline_stage

logger = get_logger("synthetic_generator")

# ─── Climate Parameters for New Delhi ────────────────────────────────────────
# Baseline values derived from historical averages
BASE_TEMP_MEAN = 25.0       # °C annual mean
TEMP_AMPLITUDE = 10.0       # °C seasonal swing
WARMING_RATE = 0.03         # °C per year (realistic warming trend)
BASE_PRECIP = 2.5           # mm/day average
PRECIP_AMPLITUDE = 3.0      # seasonal precipitation variation
BASE_HUMIDITY = 55.0        # % annual mean
HUMIDITY_AMPLITUDE = 20.0   # seasonal humidity variation
BASE_SOLAR = 18.0           # MJ/m²/day average
SOLAR_AMPLITUDE = 6.0       # seasonal solar variation
BASE_WIND = 8.0             # m/s average wind speed
WIND_AMPLITUDE = 3.0        # seasonal wind variation


def generate_synthetic_data(
    start_year: int = SYNTHETIC_START_YEAR,
    end_year: int = SYNTHETIC_END_YEAR,
    lat: float = DEFAULT_LATITUDE,
    lon: float = DEFAULT_LONGITUDE,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic daily climate data.

    Creates multi-year synthetic data with:
    - Sinusoidal seasonal patterns
    - Long-term warming trend
    - Realistic daily noise
    - Seasonal precipitation patterns
    - Monsoon simulation (June-September)

    Args:
        start_year: First year of synthetic data.
        end_year: Last year of synthetic data.
        lat: Latitude (stored as metadata).
        lon: Longitude (stored as metadata).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with synthetic daily climate observations.
    """
    log_pipeline_stage("Synthetic Data Generation")

    np.random.seed(seed)

    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)

    # Time index for trend calculation
    t = np.arange(n_days)
    years = np.array([(d - start_date).days / 365.25 for d in dates])

    # Day of year for seasonal patterns
    doy = np.array([d.timetuple().tm_yday for d in dates])
    seasonal_phase = 2 * np.pi * (doy - 15) / 365.25  # Peak in mid-January

    # ─── Temperature ───────────────────────────────────────────────────────
    temp_trend = WARMING_RATE * years
    temp_seasonal = -TEMP_AMPLITUDE * np.sin(seasonal_phase)  # Cooler in winter
    temp_noise = np.random.normal(0, 2.0, n_days)

    temperature = BASE_TEMP_MEAN + temp_trend + temp_seasonal + temp_noise
    temperature_max = temperature + np.random.uniform(5, 10, n_days)
    temperature_min = temperature - np.random.uniform(5, 10, n_days)

    # ─── Precipitation (with monsoon simulation) ──────────────────────────
    # Monsoon: June-September (days 152-273)
    monsoon_factor = np.where((doy >= 152) & (doy <= 273), 2.5, 1.0)
    precip_seasonal = PRECIP_AMPLITUDE * np.abs(np.sin(seasonal_phase + np.pi / 3))
    precip_base = BASE_PRECIP * monsoon_factor * precip_seasonal

    # Add random storm events
    storm_mask = np.random.random(n_days) < 0.05  # 5% chance of storm
    precip_noise = np.random.exponential(1.0, n_days)
    precipitation = np.where(
        storm_mask,
        precip_base * precip_noise * 3,
        precip_base * precip_noise,
    )
    precipitation = np.maximum(precipitation, 0)

    # ─── Humidity ──────────────────────────────────────────────────────────
    humidity_seasonal = HUMIDITY_AMPLITUDE * np.sin(seasonal_phase + np.pi / 2)
    humidity_noise = np.random.normal(0, 5.0, n_days)
    humidity = BASE_HUMIDITY + humidity_seasonal + humidity_noise
    humidity = np.clip(humidity, 10, 100)

    # Higher humidity during monsoon
    humidity = np.where(
        (doy >= 152) & (doy <= 273),
        humidity + 15,
        humidity,
    )
    humidity = np.clip(humidity, 10, 100)

    # ─── Solar Radiation ───────────────────────────────────────────────────
    solar_seasonal = SOLAR_AMPLITUDE * np.sin(seasonal_phase + np.pi)
    solar_noise = np.random.normal(0, 1.5, n_days)
    solar_radiation = BASE_SOLAR + solar_seasonal + solar_noise

    # Reduce solar during monsoon (cloud cover)
    solar_radiation = np.where(
        (doy >= 152) & (doy <= 273),
        solar_radiation * 0.7,
        solar_radiation,
    )
    solar_radiation = np.maximum(solar_radiation, 0)

    # ─── Wind Speed ────────────────────────────────────────────────────────
    wind_seasonal = WIND_AMPLITUDE * np.sin(seasonal_phase + np.pi / 4)
    wind_noise = np.random.normal(0, 2.0, n_days)
    wind_speed = BASE_WIND + wind_seasonal + wind_noise
    wind_speed = np.maximum(wind_speed, 0)

    # ─── Assemble DataFrame ───────────────────────────────────────────────
    df = pd.DataFrame({
        COL_DATE: dates,
        COL_TEMP_MEAN: np.round(temperature, 2),
        COL_TEMP_MAX: np.round(temperature_max, 2),
        COL_TEMP_MIN: np.round(temperature_min, 2),
        COL_PRECIPITATION: np.round(precipitation, 2),
        COL_HUMIDITY: np.round(humidity, 2),
        COL_SOLAR_RADIATION: np.round(solar_radiation, 2),
        COL_WIND_SPEED: np.round(wind_speed, 2),
    })

    logger.info(
        f"Synthetic data generated: {len(df)} days "
        f"({start_year}-{end_year}), "
        f"Lat: {lat}, Lon: {lon}"
    )
    logger.info(f"Temperature range: {df[COL_TEMP_MEAN].min():.1f}°C to {df[COL_TEMP_MEAN].max():.1f}°C")
    logger.info(f"Mean precipitation: {df[COL_PRECIPITATION].mean():.2f} mm/day")

    # Save synthetic data to raw directories
    from src.config import RAW_DIR
    from src.utils import save_csv, save_json
    raw_dir = RAW_DIR / "merged"
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_csv(df, raw_dir / "merged_raw.csv")
    save_json(df.to_dict(orient="records"), raw_dir / "merged_raw.json")

    return df


def generate_scenario(
    scenario: str = "baseline",
    start_year: int = SYNTHETIC_START_YEAR,
    end_year: int = SYNTHETIC_END_YEAR,
) -> pd.DataFrame:
    """
    Generate synthetic data for a specific climate scenario.

    Scenarios:
        - baseline: Standard warming trend
        - optimistic: Reduced warming (0.01°C/year)
        - pessimistic: Accelerated warming (0.05°C/year)

    Args:
        scenario: Scenario name.
        start_year: Start year.
        end_year: End year.

    Returns:
        DataFrame with scenario-specific synthetic data.
    """
    global WARMING_RATE

    original_rate = WARMING_RATE

    if scenario == "optimistic":
        WARMING_RATE = 0.01
    elif scenario == "pessimistic":
        WARMING_RATE = 0.05
    elif scenario == "baseline":
        WARMING_RATE = 0.03
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    logger.info(f"Generating '{scenario}' scenario (warming rate: {WARMING_RATE}°C/yr)")
    df = generate_synthetic_data(start_year=start_year, end_year=end_year)

    WARMING_RATE = original_rate
    return df
