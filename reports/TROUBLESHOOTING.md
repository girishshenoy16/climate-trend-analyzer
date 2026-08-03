# Troubleshooting Matrix & Solutions

## Common Issues & Resolutions

### Installation Issues

| Issue                                        | Cause                           | Solution                                                  |
|----------------------------------------------|---------------------------------|-----------------------------------------------------------|
| `ModuleNotFoundError: No module named 'src'` | Project root not in Python path | Run from project root or set `PYTHONPATH=.`               |
| `pip install` fails                          | Network/proxy issues            | Use `pip install --proxy <proxy_url> -r requirements.txt` |
| `matplotlib` backend error                   | No display server               | Add `matplotlib.use("Agg")` before import                 |
| `statsmodels` import error                   | Version conflict                | Run `pip install statsmodels==0.14.2`                     |

### API & Data Issues

| Issue                             | Cause                      | Solution                                    |
|-----------------------------------|----------------------------|---------------------------------------------|
| `ConnectionError: NASA POWER API` | API downtime or rate limit | Use `--simulate` flag for synthetic data    |
| `HTTPError: 429`                  | Too many requests          | Wait 60 seconds or reduce request frequency |
| `JSON decode error`               | Malformed API response     | Clear cache: `rm -rf data/cache/*`          |
| `Empty DataFrame after merge`     | Date range mismatch        | Check START_DATE and END_DATE in config.py  |

### Processing Issues

| Issue                           | Cause                      | Solution                                        |
|---------------------------------|----------------------------|-------------------------------------------------|
| `KeyError: 'date'`              | Column name mismatch       | Ensure raw data has 'date' column               |
| `ValueError: Schema validation` | Missing required columns   | Check API response has expected parameters      |
| `NaN in processed data`         | Insufficient interpolation | Increase data range or adjust imputation method |

### Visualization Issues

| Issue                  | Cause                   | Solution                                       |
|------------------------|-------------------------|------------------------------------------------|
| `Figure not saving`    | Directory doesn't exist | Run `mkdir -p outputs/figures`                 |
| `Font warnings`        | Missing system fonts    | Install default fonts or use `sns.set_theme()` |
| `Chart.js not loading` | CDN blocked             | Use local Chart.js copy in `docs/js/`          |

### Streamlit Issues

| Issue                          | Cause                      | Solution                                          |
|--------------------------------|----------------------------|---------------------------------------------------|
| `streamlit: command not found` | Not installed              | Run `pip install streamlit`                       |
| `Port 8501 in use`             | Another Streamlit instance | Kill existing process or use `--server.port 8502` |
| `Data not loading`             | Processed data missing     | Run `python main.py --simulate` first             |

### Testing Issues

| Issue                        | Cause                     | Solution                                    |
|------------------------------|---------------------------|---------------------------------------------|
| `pytest: no tests collected` | Test files not discovered | Ensure `tests/__init__.py` exists           |
| `ImportError in tests`       | Path issues               | Run `pytest` from project root              |
| `Fixture not found`          | Scope mismatch            | Check fixture decorator and scope parameter |

---

## Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set environment variable:

```bash
export LOG_LEVEL=DEBUG
python main.py --simulate
```

---

## Performance Tips

1. **Use `--simulate`** for fast iteration without API calls
2. **Reduce date range** in config.py for quicker testing
3. **Cache API responses** are stored in `data/cache/` - delete to refresh
4. **Figures are saved at 300 DPI** - modify `FIGURE_DPI` in config.py for faster rendering

---

## Getting Help

1. Check this troubleshooting guide
2. Review error messages carefully
3. Run individual pipeline steps to isolate issues
4. Check `logs/` directory for detailed log files
