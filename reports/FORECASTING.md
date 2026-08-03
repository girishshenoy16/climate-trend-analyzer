# Forecast Classification & Validation

This document explains how the Climate Trend Analyzer evaluates, classifies, and validates temperature forecasts.

---

## Overview

The forecasting pipeline uses **Holt-Winters Exponential Smoothing** as the primary model, with automatic fallback to **linear trend extrapolation** when the primary model exhibits instability. Every forecast undergoes rigorous validation before being used for reporting.

---

## Forecast Classification

Every forecast is classified into one of three categories:

| Classification | Meaning | Usage |
|---------------|---------|-------|
| **Reliable** | Model exhibits stable, consistent behavior | Use as primary projection |
| **Moderate** | Model shows some instability but remains usable | Use with caution |
| **Exploratory** | Model exhibits significant instability | Scenario analysis only; fallback used |

### Why Forecasts May Be Classified as "Exploratory"

The Holt-Winters model extrapolates seasonal patterns into the future. When the extrapolated trend diverges significantly from the historical trend, the forecast is classified as **EXPLORATORY**. This is expected behavior for:

- **Datasets with weak or no clear trend** — The model may amplify noise
- **Short time series** — Insufficient data to establish reliable seasonal patterns
- **High-variability climates** — Natural variability exceeds model capacity
- **Flat or near-flat historical trends** — Any extrapolation appears extreme relative to history

**This is not a software failure.** It is the pipeline correctly identifying that the primary model's projection should not be used at face value.

---

## Reliability Score

The reliability score (0.0 to 1.0) quantifies overall forecast confidence by combining 7 weighted components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Trend Consistency | 20% | Ratio of forecast trend to historical trend |
| In-Sample R² | 15% | How well the model fits historical data |
| RMSE Normalized | 15% | Prediction error relative to temperature range |
| Statistical Significance | 10% | p-value of forecast trend |
| Walk-Forward Validation | 20% | Out-of-sample performance across 3 folds |
| Quality Checks | 10% | Pass/fail on 6 sanity checks |
| Benchmark Comparison | 10% | Performance vs naive/seasonal naive baselines |

### Reliability Labels

| Score Range | Label | Interpretation |
|-------------|-------|----------------|
| 0.70 – 1.00 | High | Strong confidence in forecast |
| 0.50 – 0.69 | Moderate | Some concerns; use with caution |
| 0.00 – 0.49 | Low | Significant concerns; scenario only |

---

## Fallback Mechanism

When the primary Holt-Winters model is classified as EXPLORATORY:

1. **Linear trend extrapolation** is generated as an alternative
2. The fallback's **trend ratio** is compared to the primary model's trend ratio
3. If the fallback is more stable (lower trend ratio), it **replaces** the primary forecast
4. The forecast is re-classified using the fallback model's metrics

### Why Fallback Models Are Used

Holt-Winters can produce extreme extrapolations when:
- The seasonal component dominates the trend
- The historical trend is near-zero (making any extrapolation appear extreme)
- The model overfits to recent fluctuations

Linear trend extrapolation is more conservative and often better aligned with historical patterns, making it a safer choice for reporting.

---

## Walk-Forward Validation

The pipeline performs **3-fold walk-forward validation**:

- **Training window**: Minimum 2 years (730 days)
- **Test window**: 1 year (365 days)
- **Expanding window**: Each fold adds the previous test period to training

This evaluates how well the model would have performed on unseen data, providing a realistic estimate of forecast accuracy.

---

## Model Benchmarking

The primary model is benchmarked against three baselines:

| Model | Description |
|-------|-------------|
| Naive | Last value repeated |
| Seasonal Naive | Last year's values repeated |
| Linear Trend | Linear regression extrapolation |

If Holt-Winters does not outperform these baselines, the pipeline notes this in the forecast classification.

---

## Quality Checks

Six automated quality checks validate every forecast. Each check is evaluated independently and reported with its individual status:

| Check | Threshold | Description |
|-------|-----------|-------------|
| NaN Values | `nan_count == 0` | No missing values in forecast |
| Physical Limits | `-50°C ≤ values ≤ 60°C` | Forecast within reasonable temperature bounds for New Delhi |
| Slope Check | `|trend| ≤ 10.0°C/decade` | Forecast trend not extreme |
| Confidence Intervals | `invalid_ci_count == 0` | Upper bound > Lower bound for all intervals |
| Mean Deviation | `deviation ≤ 20%` | Forecast mean within 20% of historical mean |
| Trend Ratio | `ratio ≤ 6.0x` | Forecast trend within 6x of historical trend |

### Individual Check Reporting

Reports display each check with:
- **Status**: PASS or FAIL based on the specific threshold
- **Details**: Actual values (e.g., "0 NaN values", "Trend: 10.983 °C/decade")

Failed checks contribute to EXPLORATORY classification but do not prevent forecast generation.

---

## Interpreting Forecast Output

When reviewing forecast results:

1. **Check the classification** — RELIABLE forecasts can be used directly; EXPLORATORY forecasts should be treated as scenario analysis
2. **Review the reliability score** — Higher scores indicate greater confidence
3. **Examine the trend ratio** — Values closer to 1.0x indicate better alignment with historical patterns
4. **Check if fallback was used** — Linear fallback indicates the primary model was unstable
5. **Review confidence intervals** — Wider intervals indicate greater uncertainty

---

## Technical Reference

- **Primary Model**: Holt-Winters Exponential Smoothing (additive trend, multiplicative seasonal)
- **Fallback Model**: Linear trend extrapolation (scipy.stats.linregress)
- **Confidence Level**: 95% (z = 1.96)
- **Prediction Intervals**: Expanding (grow with √horizon)
- **Trend Dampening**: Applied when trend ratio > 3.0x (final safeguard)
- **Forecast Horizon**: 3 years (1,095 days)

---

*This document describes the forecasting methodology as of Version 1.0.0.*
