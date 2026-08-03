# System Architecture & Data Pipeline Guide

## Overview

The Climate Trend Analyzer follows a modular, layered architecture with clear separation of concerns across 8 distinct layers.

---

## Architecture Layers

### 1. Data Acquisition Layer
**Module:** `src/api_fetcher.py`

- Connects to NASA POWER API and Open-Meteo Climate Archive API
- Implements HTTP response caching (`data/cache/`)
- Handles retries with exponential backoff
- Saves raw JSON payloads to `data/raw/`
- Merges datasets into unified format

### 2. Data Storage Layer
**Directory:** `data/`

```
data/
├── cache/           # HTTP response cache
├── raw/
│   ├── nasa_power/  # Raw NASA POWER JSON & CSV
│   ├── open_meteo/  # Raw Open-Meteo JSON & CSV
│   └── merged/      # Merged raw dataset
└── processed/       # Clean analytical CSV
```

### 3. Data Processing Layer
**Modules:** `src/data_loader.py`, `src/preprocessing.py`

- Schema validation and type conversion
- Date bounds checking
- Linear missing value imputation
- Temporal feature engineering (Year, Month, Season, Day of Year)
- Rolling averages (7-day, 30-day, 365-day)
- Lag features (1, 7, 30 days)

### 4. Analytics & ML Layer
**Modules:** `src/eda.py`, `src/trend_analysis.py`, `src/anomaly_detection.py`, `src/forecasting.py`

- Summary statistics and correlation analysis
- Linear trend estimation (y = mx + c)
- STL seasonal decomposition
- Z-Score anomaly detection (|Z| > 2.5)
- Isolation Forest unsupervised anomaly detection
- Holt-Winters Exponential Smoothing forecasting

### 5. Executive Insights Engine
**Module:** `src/report_generator.py`

- Climate Risk Score: (0.40 × Temp Trend) + (0.30 × Rain Dev) + (0.30 × Anomaly Freq)
- Risk categories: Low (0-0.25), Moderate (0.25-0.50), High (0.50-0.75), Very High (0.75-1.00)
- Natural language insights generation
- Strategic adaptation recommendations

### 6. Visualization Layer
**Module:** `src/visualization.py`

10 publication-quality PNG figures exported at 300 DPI:
1. Temperature Trend
2. Rainfall Trend
3. Humidity Trend
4. Solar Radiation Trend
5. Correlation Heatmap
6. Seasonal Decomposition
7. Forecast Plot
8. Anomaly Detection Plot
9. Monthly Distribution
10. Correlation Matrix

### 7. Reporting Layer
**Module:** `src/report_generator.py` (markdown generators)

4 automated markdown reports:
- Executive Summary
- Technical Report
- Model Summary
- Data Quality Report

### 8. Dual Deployment Layer

**GitHub Pages (Primary):** `docs/index.html`
- Glassmorphic dark-theme UI
- Chart.js interactive charts
- Leaflet.js regional map
- Reads from `docs/data/*.json`

**Streamlit Cloud (Secondary):** `streamlit_app.py`
- Multi-page navigation
- Interactive Plotly charts
- Reads from `data/processed/` and `docs/data/`

---

## Data Flow

```
APIs → main.py → data/raw/ → preprocessing → data/processed/
                                                    |
                                        +-----------+-----------+
                                        |           |           |
                                   eda.py    trend_analysis  anomaly_detection
                                        |           |           |
                                        +-----+-----+-----+----+
                                              |           |
                                        forecasting  report_generator
                                              |           |
                                              v           v
                                        docs/data/*.json
                                              |
                                    +---------+---------+
                                    |                   |
                              GitHub Pages         Streamlit
```

---

## Configuration

All configuration is centralized in `src/config.py`:
- Geographic coordinates
- API endpoints and parameters
- Processing settings
- Forecasting parameters
- Risk score weights
- File paths

---

## Testing Strategy

- Unit tests for each source module
- Integration test for full pipeline
- pytest with fixtures for data generation
- CI/CD via GitHub Actions
