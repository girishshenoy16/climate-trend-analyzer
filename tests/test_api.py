"""Tests for API fetcher module."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.api_fetcher import NASAPowerFetcher, OpenMeteoFetcher, merge_datasets


class TestNASAPowerFetcher:
    """Tests for NASA POWER API client."""

    def test_init(self):
        fetcher = NASAPowerFetcher(lat=28.61, lon=77.21)
        assert fetcher.lat == 28.61
        assert fetcher.lon == 77.21

    @patch("src.api_fetcher.requests.get")
    def test_fetch_with_retry_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "properties": {
                "parameter": {
                    "T2M": {"20240101": 25.0, "20240102": 26.0},
                    "PRECTOTCORR": {"20240101": 2.5, "20240102": 3.0},
                    "RH2M": {"20240101": 55.0, "20240102": 60.0},
                    "ALLSKY_SFC_SW_DWN": {"20240101": 18.0, "20240102": 19.0},
                    "WS2M": {"20240101": 5.0, "20240102": 6.0},
                }
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        fetcher = NASAPowerFetcher()
        df = fetcher._parse_response(mock_resp.json())

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "temperature" in df.columns

    def test_parse_response_empty(self):
        fetcher = NASAPowerFetcher()
        with pytest.raises(ValueError):
            fetcher._parse_response({"properties": {"parameter": {}}})


class TestOpenMeteoFetcher:
    """Tests for Open-Meteo API client."""

    def test_init(self):
        fetcher = OpenMeteoFetcher(lat=28.61, lon=77.21)
        assert fetcher.lat == 28.61

    def test_parse_response(self):
        fetcher = OpenMeteoFetcher()
        data = {
            "daily": {
                "time": ["2024-01-01", "2024-01-02"],
                "temperature_2m_mean": [25.0, 26.0],
                "temperature_2m_max": [30.0, 31.0],
                "temperature_2m_min": [20.0, 21.0],
                "precipitation_sum": [2.5, 3.0],
                "relative_humidity_2m_mean": [55.0, 60.0],
                "shortwave_radiation_sum": [18.0, 19.0],
                "windspeed_10m_max": [10.0, 11.0],
            }
        }
        df = fetcher._parse_response(data)
        assert len(df) == 2
        assert "temperature" in df.columns


class TestMergeDatasets:
    """Tests for dataset merging."""

    def test_merge_basic(self):
        nasa_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "temperature": [25.0, 26.0],
            "precipitation": [2.0, 3.0],
            "humidity": [55.0, 60.0],
            "solar_radiation": [18.0, 19.0],
            "wind_speed": [5.0, 6.0],
            "temperature_max": [30.0, 31.0],
            "temperature_min": [20.0, 21.0],
        })
        openmeteo_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "temperature": [24.5, 25.5],
            "precipitation": [2.5, 3.5],
            "humidity": [56.0, 61.0],
            "solar_radiation": [17.5, 18.5],
            "wind_speed": [4.5, 5.5],
            "temperature_max": [29.5, 30.5],
            "temperature_min": [19.5, 20.5],
        })

        merged = merge_datasets(nasa_df, openmeteo_df)
        assert len(merged) == 2
        assert "date" in merged.columns
