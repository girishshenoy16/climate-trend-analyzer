# Virtual Simulation Execution Guide

## Overview

This guide explains how to run the Climate Trend Analyzer using synthetic climate data, which requires no API keys or internet connectivity.

---

## When to Use Simulation Mode

- Testing the pipeline without API access
- Demonstrating the system in air-gapped environments
- Development and debugging
- CI/CD pipeline testing

---

## Quick Start

### Step 1: Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Run in Simulation Mode

```bash
python main.py --simulate
```

This will:
1. Generate 10 years of synthetic daily climate data (2015-2024)
2. Apply sinusoidal seasonal patterns with warming trend
3. Simulate monsoon precipitation (June-September)
4. Execute the complete analysis pipeline
5. Export all outputs

### Step 3: Verify Outputs

```bash
# Check generated figures
ls outputs/figures/

# Check JSON feeds
ls docs/data/

# Check reports
ls outputs/reports/

# Check processed data
ls data/processed/
```

---

## Synthetic Data Parameters

The synthetic generator produces realistic climate data for New Delhi, India:

| Parameter            | Value            |
|----------------------|------------------|
| Location             | 28.61°N, 77.21°E |
| Baseline Temperature | 25.0°C           |
| Warming Rate         | 0.03°C/year      |
| Seasonal Amplitude   | 10.0°C           |
| Mean Precipitation   | 2.5 mm/day       |
| Monsoon Period       | June-September   |

### Climate Scenarios

```python
from src.synthetic_generator import generate_scenario

# Baseline (default)
df = generate_scenario("baseline")

# Optimistic (reduced warming)
df = generate_scenario("optimistic")

# Pessimistic (accelerated warming)
df = generate_scenario("pessimistic")
```

---

## Viewing Results

### GitHub Pages Dashboard
```bash
start docs/index.html
```

### Streamlit Interactive Dashboard
```bash
streamlit run streamlit_app.py
```

### Static Figures
Open any PNG from `outputs/figures/` in an image viewer.

---

## Troubleshooting

| Issue               | Solution                                  |
|---------------------|-------------------------------------------|
| ModuleNotFoundError | Ensure virtual environment is activated   |
| PermissionError     | Check file permissions in data/ directory |
| Plotting errors     | Run `pip install matplotlib`              |
| Streamlit not found | Run `pip install streamlit`               |

---

## CI/CD Simulation

The GitHub Actions workflow runs simulation mode automatically:

```yaml
- name: Verify pipeline execution
  run: python main.py --simulate
```
