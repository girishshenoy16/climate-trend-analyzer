"""
Structured Production Logging Module for Climate Trend Analyzer.

Provides configured loggers for both console and file output
with consistent formatting, performance timing, memory tracking,
and statistical warning utilities across all pipeline modules.
"""

import logging
import sys
import time
import tracemalloc
from datetime import datetime

from src.config import PROJECT_ROOT


LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ─── Logger Factory ───────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a configured logger instance.

    Args:
        name: Logger name (typically module name).
        level: Logging level (default: INFO).

    Returns:
        Configured logger with console and file handlers.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"pipeline_{today}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ─── Pipeline Stage Logging ───────────────────────────────────────────────────

def log_pipeline_stage(stage_name: str) -> None:
    """Log a pipeline stage separator for readability (single-line format)."""
    logger = get_logger("pipeline")
    logger.info(f"--- {stage_name} ---")


# ─── Performance Timing ──────────────────────────────────────────────────────

class PipelineTimer:
    """
    Tracks pipeline execution timing, memory usage, and CPU time.

    Usage:
        timer = PipelineTimer()
        timer.start()
        # ... pipeline work ...
        timer.stop()
        timer.log_summary()
    """

    def __init__(self) -> None:
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._stage_times: dict[str, float] = {}
        self._stage_start: float = 0.0
        self._current_stage: str = ""
        self._peak_memory_bytes: int = 0

    def start(self) -> None:
        """Start the pipeline timer and memory tracking."""
        self._start_time = time.perf_counter()
        tracemalloc.start()
        logger = get_logger("pipeline")
        logger.info("Pipeline execution started")

    def stop(self) -> None:
        """Stop the pipeline timer and capture peak memory."""
        self._end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        self._peak_memory_bytes = peak
        tracemalloc.stop()

    def begin_stage(self, stage_name: str) -> None:
        """Mark the beginning of a pipeline stage."""
        if self._current_stage and self._current_stage in self._stage_times:
            elapsed = time.perf_counter() - self._stage_start
            self._stage_times[self._current_stage] = elapsed

        self._current_stage = stage_name
        self._stage_start = time.perf_counter()

    def end_stage(self) -> None:
        """Mark the end of the current pipeline stage."""
        if self._current_stage:
            elapsed = time.perf_counter() - self._stage_start
            self._stage_times[self._current_stage] = elapsed

    def log_summary(self) -> None:
        """Log the complete timing and memory summary."""
        logger = get_logger("pipeline")
        total = self._end_time - self._start_time

        # Finalize any pending stage
        if self._current_stage and self._current_stage not in self._stage_times:
            elapsed = time.perf_counter() - self._stage_start
            self._stage_times[self._current_stage] = elapsed

        logger.info("=" * 60)
        logger.info("  PIPELINE PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Total Execution Time: {total:.2f}s")
        logger.info(f"  Peak Memory Usage: {self._format_memory(self._peak_memory_bytes)}")

        if self._stage_times:
            logger.info("  Stage Breakdown:")
            for stage, elapsed in self._stage_times.items():
                pct = (elapsed / total * 100) if total > 0 else 0
                logger.info(f"    {stage}: {elapsed:.2f}s ({pct:.1f}%)")

        logger.info("=" * 60)

    @property
    def total_elapsed(self) -> float:
        """Return total elapsed time in seconds."""
        return self._end_time - self._start_time

    @property
    def peak_memory_mb(self) -> float:
        """Return peak memory usage in MB."""
        return self._peak_memory_bytes / (1024 * 1024)

    @staticmethod
    def _format_memory(bytes_val: int) -> str:
        """Format byte value to human-readable string."""
        mb = bytes_val / (1024 * 1024)
        if mb < 1024:
            return f"{mb:.1f} MB"
        gb = mb / 1024
        return f"{gb:.2f} GB"


# ─── Statistical Warning Utilities ───────────────────────────────────────────

# Warning categories with severity, interpretation, and recommended action
WARNING_CATEGORIES = {
    # Data Quality Warnings
    "DATA_QUALITY": {
        "category": "Data Quality",
        "severity": "medium",
        "interpretation": "The dataset contains quality issues that may affect analysis reliability.",
        "action": "Review data collection process and consider additional validation."
    },
    "MISSING_VALUES": {
        "category": "Data Quality",
        "severity": "low",
        "interpretation": "Missing values were imputed, which may introduce bias or reduce variability.",
        "action": "Consider collecting additional data or using alternative imputation methods."
    },
    "OUTLIERS_DETECTED": {
        "category": "Data Quality",
        "severity": "medium",
        "interpretation": "Statistical outliers were detected that may influence results.",
        "action": "Investigate outliers for data entry errors or genuine extreme events."
    },
    # Statistical Warnings
    "NON_SIGNIFICANT": {
        "category": "Statistical",
        "severity": "medium",
        "interpretation": "The observed trend may be due to random variation rather than a true climate signal.",
        "action": "Consider increasing analysis period or using alternative trend methods."
    },
    "LOW_R2": {
        "category": "Statistical",
        "severity": "medium",
        "interpretation": "The linear model explains very little of the observed variation.",
        "action": "Examine for non-linear patterns, seasonality, or structural breaks."
    },
    "HIGH_P_VALUE": {
        "category": "Statistical",
        "severity": "low",
        "interpretation": "The p-value indicates insufficient evidence to reject the null hypothesis.",
        "action": "Consider increasing sample size or using more sensitive statistical tests."
    },
    # Forecast Warnings — Expected analytical outcomes, not software failures
    "TREND_DIVERGENCE": {
        "category": "Forecast",
        "severity": "info",
        "interpretation": (
            "Forecast trend diverges significantly from historical trend. "
            "This is expected when the primary model (Holt-Winters) extrapolates differently "
            "than the underlying climate pattern. The pipeline automatically falls back "
            "to a more stable linear trend forecast."
        ),
        "action": (
            "No action required — pipeline handles this automatically. "
            "Fallback forecast is used for reporting."
        )
    },
    "TREND_DIVERGENCE_MINOR": {
        "category": "Forecast",
        "severity": "info",
        "interpretation": (
            "Moderate deviation between historical and forecast trends. "
            "This is within normal range for time-series extrapolation."
        ),
        "action": "Forecast is usable; no fallback required."
    },
    "LOW_FORECAST_R2": {
        "category": "Forecast",
        "severity": "info",
        "interpretation": (
            "Forecast trend explains limited variance (low R²). "
            "This is expected for datasets with weak or no clear trend. "
            "The forecast remains valid but should be interpreted with appropriate caution."
        ),
        "action": (
            "No action required — classify forecast as 'exploratory' "
            "and recommend caution in usage."
        )
    },
    "HIGH_RMSE": {
        "category": "Forecast",
        "severity": "info",
        "interpretation": (
            "Model prediction errors (RMSE) are relatively large. "
            "This is expected for datasets with high natural variability."
        ),
        "action": "Forecast is usable; confidence intervals reflect uncertainty."
    },
    "DAMPENED_FORECAST": {
        "category": "Forecast",
        "severity": "info",
        "interpretation": (
            "Forecast was dampened as a final safeguard to prevent unrealistic extrapolation. "
            "This is expected behavior when trend ratio exceeds threshold."
        ),
        "action": "Dampened forecast is used for reporting."
    },
    "FALLBACK_FORECAST": {
        "category": "Forecast",
        "severity": "info",
        "interpretation": (
            "Primary model (Holt-Winters) was classified as EXPLORATORY. "
            "Linear trend extrapolation was selected as a more stable alternative. "
            "This is expected behavior for datasets where Holt-Winters exhibits instability."
        ),
        "action": "Linear trend forecast is used for reporting."
    },
    # Model Assumption Warnings
    "MODEL_ASSUMPTION": {
        "category": "Model Assumption",
        "severity": "medium",
        "interpretation": "Model assumptions may be violated, affecting result validity.",
        "action": "Review model assumptions and consider alternative approaches."
    },
    "HETEROSCEDASTICITY": {
        "category": "Model Assumption",
        "severity": "medium",
        "interpretation": "Non-constant variance in residuals may affect confidence intervals.",
        "action": "Consider weighted regression or robust standard errors."
    },
    "AUTOCORRELATION": {
        "category": "Model Assumption",
        "severity": "medium",
        "interpretation": "Residual autocorrelation may affect statistical inference.",
        "action": "Consider time series models that account for autocorrelation."
    },
}


def warn_statistical_issue(
    issue_type: str,
    details: str,
    logger_name: str = "pipeline",
    include_interpretation: bool = True,
) -> dict:
    """
    Log a standardized statistical warning with categorization.

    Args:
        issue_type: Category of the issue (e.g., 'LOW_R2', 'NON_SIGNIFICANT').
        details: Human-readable description of the issue.
        logger_name: Logger to use.
        include_interpretation: Whether to include interpretation and action.

    Returns:
        Dictionary with warning metadata for report integration.
    """
    logger = get_logger(logger_name)
    category = WARNING_CATEGORIES.get(issue_type, {})
    severity = category.get("severity", "unknown")
    warning_category = category.get("category", "Unknown")
    interpretation = category.get("interpretation", "")
    action = category.get("action", "")

    log_msg = f"[{warning_category}] [{severity.upper()}] [{issue_type}]: {details}"
    if include_interpretation and interpretation:
        log_msg += f" | Interpretation: {interpretation}"

    if severity == "info":
        logger.info(log_msg)
    else:
        logger.warning(log_msg)

    return {
        "type": issue_type,
        "category": warning_category,
        "severity": severity,
        "details": details,
        "interpretation": interpretation,
        "recommended_action": action
    }


def check_trend_significance(
    p_value: float,
    r_squared: float,
    variable_name: str,
    logger_name: str = "trend_analysis",
) -> list[str]:
    """
    Check and log warnings for trend significance issues.

    Args:
        p_value: P-value from trend regression.
        r_squared: R-squared from trend regression.
        variable_name: Name of the variable being analyzed.
        logger_name: Logger to use.

    Returns:
        List of warning message strings (empty if no issues).
    """
    warnings = []

    if p_value > 0.05:
        msg = (
            f"{variable_name} trend is not statistically significant "
            f"(p={p_value:.4f} > 0.05). Trend may be due to random variation."
        )
        warn_statistical_issue("NON_SIGNIFICANT", msg, logger_name)
        warnings.append(msg)

    if r_squared < 0.01:
        msg = (
            f"{variable_name} trend explains very little variance "
            f"(R²={r_squared:.6f}). Linear model may be inappropriate."
        )
        warn_statistical_issue("LOW_R2", msg, logger_name)
        warnings.append(msg)

    return warnings


def validate_forecast_consistency(
    historical_trend: float,
    forecast_trend: float,
    variable_name: str = "Temperature",
    logger_name: str = "pipeline",
) -> list[str]:
    """
    Validate consistency between historical and forecast trends.

    Args:
        historical_trend: Historical warming rate (°C/decade).
        forecast_trend: Forecast warming rate (°C/decade).
        variable_name: Name of the variable.
        logger_name: Logger to use.

    Returns:
        List of warning message strings (empty if consistent).
    """
    warnings = []

    if historical_trend == 0:
        return warnings

    ratio = forecast_trend / historical_trend
    pct_diff = abs(ratio - 1.0) * 100

    if pct_diff > 100:
        msg = (
            f"Significant divergence between historical ({historical_trend:.3f} °C/decade) "
            f"and forecast ({forecast_trend:.3f} °C/decade) {variable_name.lower()} trends. "
            f"Difference: {pct_diff:.1f}%. This may indicate model extrapolation issues."
        )
        warn_statistical_issue("TREND_DIVERGENCE", msg, logger_name)
        warnings.append(msg)
    elif pct_diff > 50:
        msg = (
            f"Moderate divergence between historical ({historical_trend:.3f} °C/decade) "
            f"and forecast ({forecast_trend:.3f} °C/decade) {variable_name.lower()} trends. "
            f"Difference: {pct_diff:.1f}%."
        )
        warn_statistical_issue("TREND_DIVERGENCE_MINOR", msg, logger_name)
        warnings.append(msg)

    return warnings


def log_error(module: str, error: Exception, context: str = "") -> None:
    """Log an error with context information."""
    logger = get_logger(module)
    msg = f"Error: {type(error).__name__}: {error}"
    if context:
        msg = f"{context} | {msg}"
    logger.error(msg, exc_info=True)
