"""
Static High-Resolution Plot Exporter for Climate Trend Analyzer.

Generates 10 publication-quality PNG figures covering temperature trends,
rainfall analysis, humidity, solar radiation, correlation heatmaps,
seasonal decomposition, forecasts, anomaly detection, and distributions.
Optimized for performance with reusable figure styles, cached computations,
and proper cleanup.
"""

from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd

from src.config import (
    FIGURE_BBOX_INCHES,
    FIGURE_DPI,
    FIGURE_FORMAT,
    FIGURES_DIR,
)
from src.constants import (
    CHART_COLORS,
    COL_ANOMALY_COMBINED,
    COL_DATE,
    COL_HUMIDITY,
    COL_MONTH,
    COL_PRECIPITATION,
    COL_RESIDUAL,
    COL_SEASONAL,
    COL_SOLAR_RADIATION,
    COL_TEMP_MEAN,
    COL_TREND,
    COLORS,
    UNITS,
)
from src.logger import get_logger

logger = get_logger("visualization")

# ─── Apply consistent styling once ────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams.update({
    "figure.facecolor": COLORS["background"],
    "axes.facecolor": COLORS["card"],
    "axes.edgecolor": COLORS["grid"],
    "axes.labelcolor": COLORS["text"],
    "text.color": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "grid.color": COLORS["grid"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.max_open_warning": 0,  # Suppress warning for multiple figures
})

# ─── Cached formatters and locators ───────────────────────────────────────────
_DATE_FORMATTER = mdates.DateFormatter("%Y")
_YEAR_LOCATOR = mdates.YearLocator()
_MONTHS_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# ─── Rolling average cache (avoids redundant computations) ───────────────────
_rolling_cache: dict[tuple[str, int], pd.Series] = {}


def _save_figure(fig: plt.Figure, name: str) -> Path:
    """Save figure to outputs/figures/ with standard settings and close."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = FIGURES_DIR / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(
        filepath, dpi=FIGURE_DPI, bbox_inches=FIGURE_BBOX_INCHES,
        facecolor=fig.get_facecolor(), edgecolor="none",
    )
    plt.close(fig)  # Explicitly close to free memory
    logger.info(f"Figure saved: {filepath}")
    return filepath


def _get_date_formatter():
    """Return a reusable date formatter."""
    return _DATE_FORMATTER


def _apply_date_axis(ax: plt.Axes) -> None:
    """Apply consistent date formatting to an axis."""
    ax.xaxis.set_major_formatter(_DATE_FORMATTER)
    ax.xaxis.set_major_locator(_YEAR_LOCATOR)


def _compute_rolling(series: pd.Series, window: int = 30) -> pd.Series:
    """Compute rolling average with caching to avoid redundant computations."""
    cache_key = (series.name or str(id(series)), window)
    if cache_key not in _rolling_cache:
        _rolling_cache[cache_key] = series.rolling(window, min_periods=1).mean()
    return _rolling_cache[cache_key]


def plot_temperature_trend(df: pd.DataFrame, trend_line: np.ndarray = None) -> Path:
    """01 - Temperature Trend with optional linear trend line."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        df[COL_DATE], df[COL_TEMP_MEAN], color=CHART_COLORS[0],
        alpha=0.6, linewidth=0.8, label="Daily Mean",
    )

    # 30-day rolling average
    if COL_TEMP_MEAN in df.columns:
        ma30 = _compute_rolling(df[COL_TEMP_MEAN], 30)
        ax.plot(
            df[COL_DATE], ma30, color=CHART_COLORS[1],
            linewidth=2, label="30-Day Moving Average",
        )

    if trend_line is not None:
        ax.plot(
            df[COL_DATE], trend_line, color=CHART_COLORS[3],
            linewidth=2.5, linestyle="--", label="Linear Trend",
        )

    ax.set_title("Historical Temperature Trend (2015-2024)", fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Temperature ({UNITS[COL_TEMP_MEAN]})")
    ax.legend(loc="upper left", framealpha=0.8)
    _apply_date_axis(ax)
    fig.autofmt_xdate()

    return _save_figure(fig, "01_temperature_trend")


def plot_rainfall_trend(df: pd.DataFrame) -> Path:
    """02 - Rainfall Trend Analysis."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

    ax1.bar(
        df[COL_DATE], df[COL_PRECIPITATION], color=CHART_COLORS[1],
        alpha=0.4, width=1, label="Daily Precipitation",
    )

    ma30 = _compute_rolling(df[COL_PRECIPITATION], 30)
    ax1.plot(
        df[COL_DATE], ma30, color=CHART_COLORS[3],
        linewidth=2, label="30-Day Moving Average",
    )

    ax1.set_title("Precipitation Trend Analysis (2015-2024)", fontweight="bold", pad=15)
    ax1.set_ylabel(f"Precipitation ({UNITS[COL_PRECIPITATION]})")
    ax1.legend(loc="upper right", framealpha=0.8)

    if COL_MONTH in df.columns:
        monthly_precip = df.groupby(COL_MONTH)[COL_PRECIPITATION].sum()
        x_pos = range(len(monthly_precip))
        ax2.bar(x_pos, monthly_precip.values, color=CHART_COLORS[1], alpha=0.7)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(
            [_MONTHS_LABELS[i] for i in monthly_precip.index - 1], rotation=45
        )
        ax2.set_ylabel("Total (mm)")
        ax2.set_title("Monthly Precipitation Totals", fontweight="bold", pad=10)

    fig.tight_layout()
    return _save_figure(fig, "02_rainfall_trend")


def plot_humidity_trend(df: pd.DataFrame) -> Path:
    """03 - Humidity Trend Analysis."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        df[COL_DATE], df[COL_HUMIDITY], color=CHART_COLORS[4],
        alpha=0.5, linewidth=0.8, label="Daily Humidity",
    )

    ma30 = _compute_rolling(df[COL_HUMIDITY], 30)
    ax.plot(
        df[COL_DATE], ma30, color=CHART_COLORS[2],
        linewidth=2, label="30-Day Moving Average",
    )

    ax.set_title("Relative Humidity Trend (2015-2024)", fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Humidity ({UNITS[COL_HUMIDITY]})")
    ax.legend(loc="upper right", framealpha=0.8)
    _apply_date_axis(ax)
    fig.autofmt_xdate()

    return _save_figure(fig, "03_humidity_trend")


def plot_solar_radiation_trend(df: pd.DataFrame) -> Path:
    """04 - Solar Radiation Trend Analysis."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        df[COL_DATE], df[COL_SOLAR_RADIATION], color=CHART_COLORS[3],
        alpha=0.5, linewidth=0.8, label="Daily Solar Radiation",
    )

    ma30 = _compute_rolling(df[COL_SOLAR_RADIATION], 30)
    ax.plot(
        df[COL_DATE], ma30, color=CHART_COLORS[0],
        linewidth=2, label="30-Day Moving Average",
    )

    ax.set_title("Solar Radiation Trend (2015-2024)", fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Solar Radiation ({UNITS[COL_SOLAR_RADIATION]})")
    ax.legend(loc="upper right", framealpha=0.8)
    _apply_date_axis(ax)
    fig.autofmt_xdate()

    return _save_figure(fig, "04_solar_radiation_trend")


def _plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    name: str,
    cmap: str = "RdYlBu_r",
    title: str = "Climate Variable Correlation Matrix",
) -> Path:
    """Shared correlation heatmap generator."""
    fig, ax = plt.subplots(figsize=(10, 8))

    labels = [col.replace("_", " ").title() for col in corr_matrix.columns]

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap=cmap, center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax,
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
    )

    ax.set_title(title, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    return _save_figure(fig, name)


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> Path:
    """05 - Correlation Heatmap."""
    return _plot_correlation_heatmap(
        corr_matrix, "05_correlation_heatmap",
        cmap="RdYlBu_r", title="Climate Variable Correlation Matrix",
    )


def plot_seasonal_decomposition(
    df: pd.DataFrame,
    trend_component: pd.Series = None,
    seasonal_component: pd.Series = None,
    residual_component: pd.Series = None,
) -> Path:
    """06 - Seasonal Decomposition (STL)."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(
        df[COL_DATE], df[COL_TEMP_MEAN], color=CHART_COLORS[0], linewidth=0.8
    )
    axes[0].set_title("Observed Temperature", fontweight="bold")
    axes[0].set_ylabel(f"Temp ({UNITS[COL_TEMP_MEAN]})")

    if trend_component is not None:
        axes[1].plot(
            df[COL_DATE][: len(trend_component)],
            trend_component.values,
            color=CHART_COLORS[1], linewidth=2,
        )
    axes[1].set_title("Trend Component", fontweight="bold")
    axes[1].set_ylabel(f"Temp ({UNITS[COL_TEMP_MEAN]})")

    if seasonal_component is not None:
        axes[2].plot(
            df[COL_DATE][: len(seasonal_component)],
            seasonal_component.values,
            color=CHART_COLORS[2], linewidth=0.8,
        )
    axes[2].set_title("Seasonal Component", fontweight="bold")
    axes[2].set_ylabel(f"Temp ({UNITS[COL_TEMP_MEAN]})")

    if residual_component is not None:
        axes[3].plot(
            df[COL_DATE][: len(residual_component)],
            residual_component.values,
            color=CHART_COLORS[3], linewidth=0.8,
        )
    axes[3].set_title("Residual Component", fontweight="bold")
    axes[3].set_ylabel(f"Temp ({UNITS[COL_TEMP_MEAN]})")
    axes[3].set_xlabel("Date")

    for ax in axes:
        _apply_date_axis(ax)

    fig.suptitle(
        "STL Seasonal Decomposition of Temperature", fontsize=16,
        fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    return _save_figure(fig, "06_seasonal_decomposition")


def plot_forecast(
    df: pd.DataFrame,
    forecast_df: pd.DataFrame = None,
) -> Path:
    """07 - Temperature Forecast with Confidence Bounds."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        df[COL_DATE], df[COL_TEMP_MEAN], color=CHART_COLORS[0],
        linewidth=0.8, alpha=0.6, label="Historical",
    )

    ma365 = _compute_rolling(df[COL_TEMP_MEAN], 365)
    ax.plot(
        df[COL_DATE], ma365, color=CHART_COLORS[1],
        linewidth=2, label="365-Day MA",
    )

    if forecast_df is not None:
        ax.plot(
            forecast_df[COL_DATE], forecast_df["forecast"],
            color=CHART_COLORS[3], linewidth=2, label="Forecast",
        )
        ax.fill_between(
            forecast_df[COL_DATE],
            forecast_df["forecast_lower"],
            forecast_df["forecast_upper"],
            color=CHART_COLORS[3], alpha=0.15, label="95% Confidence",
        )

    ax.axvline(
        x=df[COL_DATE].max(), color="white", linestyle="--",
        alpha=0.5, label="Forecast Start",
    )
    ax.set_title(
        "Temperature Forecast - Holt-Winters (3-Year Projection)",
        fontweight="bold", pad=15,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Temperature ({UNITS[COL_TEMP_MEAN]})")
    ax.legend(loc="upper left", framealpha=0.8)
    _apply_date_axis(ax)
    fig.autofmt_xdate()

    return _save_figure(fig, "07_forecast_plot")


def plot_anomaly_detection(df: pd.DataFrame) -> Path:
    """08 - Anomaly Detection Visualization."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Single scatter call with color array for better performance
    colors = np.where(df[COL_ANOMALY_COMBINED], CHART_COLORS[0], CHART_COLORS[1])
    alphas = np.where(df[COL_ANOMALY_COMBINED], 0.85, 0.3)
    sizes = np.where(df[COL_ANOMALY_COMBINED], 25, 8)

    ax.scatter(
        df[COL_DATE], df[COL_TEMP_MEAN],
        c=colors, alpha=alphas, s=sizes,
        label="Normal/Anomaly", edgecolors="none",
    )

    ma30 = _compute_rolling(df[COL_TEMP_MEAN], 30)
    ax.plot(
        df[COL_DATE], ma30, color=CHART_COLORS[3],
        linewidth=2, label="30-Day MA",
    )

    mean_temp = df[COL_TEMP_MEAN].mean()
    std_temp = df[COL_TEMP_MEAN].std()
    ax.axhline(
        y=mean_temp + 2.5 * std_temp, color=CHART_COLORS[0],
        linestyle="--", alpha=0.5, label="Upper Threshold (+2.5σ)",
    )
    ax.axhline(
        y=mean_temp - 2.5 * std_temp, color=CHART_COLORS[4],
        linestyle="--", alpha=0.5, label="Lower Threshold (-2.5σ)",
    )

    ax.set_title("Climate Anomaly Detection Results", fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Temperature ({UNITS[COL_TEMP_MEAN]})")
    ax.legend(loc="upper left", framealpha=0.8, fontsize=9)
    _apply_date_axis(ax)
    fig.autofmt_xdate()

    return _save_figure(fig, "08_anomaly_detection_plot")


def plot_monthly_distribution(df: pd.DataFrame) -> Path:
    """09 - Monthly Climate Distribution (Box Plot)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Pre-group by month for all variables
    month_groups = df.groupby(COL_MONTH)

    if COL_TEMP_MEAN in df.columns:
        month_data = [month_groups.get_group(m)[COL_TEMP_MEAN].values for m in range(1, 13)]
        bp = axes[0].boxplot(month_data, patch_artist=True, widths=0.7)
        for patch in bp["boxes"]:
            patch.set_facecolor(CHART_COLORS[0])
            patch.set_alpha(0.6)
        axes[0].set_xticklabels(_MONTHS_LABELS, rotation=45)
        axes[0].set_title("Temperature Distribution", fontweight="bold")
        axes[0].set_ylabel(f"Temp ({UNITS[COL_TEMP_MEAN]})")

    if COL_PRECIPITATION in df.columns:
        month_data = [month_groups.get_group(m)[COL_PRECIPITATION].values for m in range(1, 13)]
        bp = axes[1].boxplot(month_data, patch_artist=True, widths=0.7)
        for patch in bp["boxes"]:
            patch.set_facecolor(CHART_COLORS[1])
            patch.set_alpha(0.6)
        axes[1].set_xticklabels(_MONTHS_LABELS, rotation=45)
        axes[1].set_title("Precipitation Distribution", fontweight="bold")
        axes[1].set_ylabel(f"Precip ({UNITS[COL_PRECIPITATION]})")

    if COL_HUMIDITY in df.columns:
        month_data = [month_groups.get_group(m)[COL_HUMIDITY].values for m in range(1, 13)]
        bp = axes[2].boxplot(month_data, patch_artist=True, widths=0.7)
        for patch in bp["boxes"]:
            patch.set_facecolor(CHART_COLORS[4])
            patch.set_alpha(0.6)
        axes[2].set_xticklabels(_MONTHS_LABELS, rotation=45)
        axes[2].set_title("Humidity Distribution", fontweight="bold")
        axes[2].set_ylabel(f"Humidity ({UNITS[COL_HUMIDITY]})")

    fig.suptitle(
        "Monthly Climate Distributions", fontsize=16,
        fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    return _save_figure(fig, "09_monthly_climate_distribution")


def plot_climate_correlation_matrix(corr_matrix: pd.DataFrame) -> Path:
    """10 - Full Climate Correlation Matrix (upper triangle)."""
    return _plot_correlation_heatmap(
        corr_matrix, "10_climate_correlation_matrix",
        cmap="coolwarm", title="Climate Variable Correlation Matrix",
    )


def generate_all_figures(
    df: pd.DataFrame,
    eda_results: dict = None,
    trend_results: dict = None,
    forecast_results: dict = None,
) -> list[Path]:
    """
    Generate all 10 static visualization figures.

    Args:
        df: Processed DataFrame with all features.
        eda_results: EDA pipeline results.
        trend_results: Trend analysis results.
        forecast_results: Forecasting results.

    Returns:
        List of file paths to generated figures.
    """
    figure_paths = []

    # Clear rolling cache for fresh pipeline run
    _rolling_cache.clear()

    # Extract components for seasonal decomposition
    trend_line = None
    trend_comp = None
    seasonal_comp = None
    residual_comp = None
    corr_matrix = None
    forecast_df = None

    if trend_results and "temperature" in trend_results:
        trend_line = trend_results["temperature"].get("trend_line")
        stl = trend_results["temperature"].get("stl_components")
        if stl:
            trend_comp = stl.get(COL_TREND)
            seasonal_comp = stl.get(COL_SEASONAL)
            residual_comp = stl.get(COL_RESIDUAL)

    if eda_results:
        corr_matrix = eda_results.get("correlation_matrix")

    if forecast_results:
        forecast_df = forecast_results.get("forecast_df")

    # Generate figures sequentially (matplotlib not thread-safe)
    figure_paths.append(plot_temperature_trend(df, trend_line))
    figure_paths.append(plot_rainfall_trend(df))
    figure_paths.append(plot_humidity_trend(df))
    figure_paths.append(plot_solar_radiation_trend(df))

    if corr_matrix is not None and not corr_matrix.empty:
        figure_paths.append(plot_correlation_heatmap(corr_matrix))

    if trend_comp is not None:
        figure_paths.append(
            plot_seasonal_decomposition(df, trend_comp, seasonal_comp, residual_comp)
        )

    figure_paths.append(plot_forecast(df, forecast_df))
    figure_paths.append(plot_anomaly_detection(df))
    figure_paths.append(plot_monthly_distribution(df))

    if corr_matrix is not None and not corr_matrix.empty:
        figure_paths.append(plot_climate_correlation_matrix(corr_matrix))

    logger.info(f"Generated {len(figure_paths)} figures in {FIGURES_DIR}")
    return figure_paths
