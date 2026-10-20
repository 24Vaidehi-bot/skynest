"""
weather_api.py
--------------
Open-Meteo Forecast API integration for SkyNest.

Uses the free Open-Meteo API (no API key required for non-commercial use).
Endpoint: https://api.open-meteo.com/v1/forecast
Timezone: Asia/Kolkata (IST) for Indian airports.

Returns hourly temperature, precipitation, wind speed, and visibility
for a given latitude, longitude, and date.
"""

import requests
from datetime import datetime, timedelta
from typing import Optional
import streamlit as st

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Kolkata"
REQUEST_TIMEOUT = 10

HOURLY_VARIABLES = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "visibility",
    "weather_code",
]

WMO_DESCRIPTIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ slight hail", 99: "Thunderstorm w/ heavy hail",
}


def _wmo_to_description(code):
    if code is None:
        return "Unknown"
    return WMO_DESCRIPTIONS.get(int(code), f"WMO code {code}")


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather_forecast(latitude, longitude, forecast_date, departure_hour=8):
    """
    Fetch hourly weather forecast from Open-Meteo for an Indian airport.
    Returns a dict with weather metrics or an error message.
    """
    result = {
        "success": False,
        "error": None,
        "temperature_c": None,
        "precipitation_mm": None,
        "wind_speed_kmh": None,
        "visibility_km": None,
        "weather_description": "Unknown",
        "forecast_date": forecast_date,
        "departure_hour": departure_hour,
        "source": "Open-Meteo (open-meteo.com)",
    }

    try:
        target_dt = datetime.strptime(forecast_date, "%Y-%m-%d")
    except ValueError:
        result["error"] = f"Invalid date format: {forecast_date!r}. Expected YYYY-MM-DD."
        return result

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    delta_days = (target_dt - today).days

    if delta_days < 0:
        result["error"] = (
            f"Open-Meteo only provides forecasts for today and future dates. "
            f"The selected date {forecast_date} is in the past. "
            "Historical weather is not available via the free forecast endpoint."
        )
        return result

    if delta_days > 16:
        result["error"] = (
            f"Open-Meteo forecasts are available up to 16 days ahead. "
            f"The selected date {forecast_date} is {delta_days} days from today."
        )
        return result

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": TIMEZONE,
        "start_date": forecast_date,
        "end_date": forecast_date,
        "wind_speed_unit": "kmh",
    }

    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        result["error"] = "Weather API request timed out. Please try again."
        return result
    except requests.exceptions.ConnectionError:
        result["error"] = "Cannot connect to Open-Meteo API. Check your internet connection."
        return result
    except requests.exceptions.HTTPError as e:
        result["error"] = f"HTTP error from Open-Meteo: {e}"
        return result
    except Exception as e:
        result["error"] = f"Unexpected error fetching weather: {e}"
        return result

    hourly = data.get("hourly", {})
    time_list = hourly.get("time", [])
    if not time_list:
        result["error"] = "Open-Meteo returned no hourly data for this location/date."
        return result

    target_time_str = f"{forecast_date}T{departure_hour:02d}:00"
    idx = time_list.index(target_time_str) if target_time_str in time_list else 0

    def _safe_get(key):
        vals = hourly.get(key, [])
        if idx < len(vals):
            v = vals[idx]
            return None if v is None else float(v)
        return None

    temp = _safe_get("temperature_2m")
    precip = _safe_get("precipitation")
    wind = _safe_get("wind_speed_10m")
    vis_raw = _safe_get("visibility")
    wcode_raw = _safe_get("weather_code")
    vis_km = round(vis_raw / 1000.0, 1) if vis_raw is not None else None
    wcode = int(wcode_raw) if wcode_raw is not None else None

    result.update({
        "success": True,
        "temperature_c": round(temp, 1) if temp is not None else None,
        "precipitation_mm": round(precip, 1) if precip is not None else None,
        "wind_speed_kmh": round(wind, 1) if wind is not None else None,
        "visibility_km": vis_km,
        "weather_description": _wmo_to_description(wcode),
    })
    return result


def get_airport_coordinates_from_df(df, iata_code):
    """
    Look up (latitude, longitude) for a given IATA code from the dataset.
    Checks origin columns first, then destination columns.
    Returns (lat, lon) tuple or None if not found.
    """
    row = df[df["origin_iata"] == iata_code]
    if not row.empty:
        lat = row.iloc[0]["origin_latitude"]
        lon = row.iloc[0]["origin_longitude"]
        if lat is not None and lon is not None:
            return (float(lat), float(lon))
    row = df[df["destination_iata"] == iata_code]
    if not row.empty:
        lat = row.iloc[0]["destination_latitude"]
        lon = row.iloc[0]["destination_longitude"]
        if lat is not None and lon is not None:
            return (float(lat), float(lon))
    return None
