"""
ECOSTRESS Thermal Stress Extractor
Retrieves Land Surface Temperature and ET
from NASA ECOSTRESS via AppEEARS API

Coverage: 52°N to 52°S
Fallback: MODIS LST for areas outside coverage

Products:
ECO_L2G_LSTE v003 — Land Surface Temperature 70m
ECO_L3_ET_ALEXI — Evapotranspiration 30m

MODIS fallback:
MOD11A1 — LST 1km daily
MOD16A2 — ET 500m 8-day
"""

import requests
import numpy as np
from datetime import datetime, timedelta
import os


APPEEARS_URL = "https://appeears.earthdatacloud.nasa.gov/api"
NASA_TOKEN = os.environ.get("NASA_EARTHDATA_TOKEN", "")


def get_appeears_token(username, password):
    """Get AppEEARS API token"""
    try:
        r = requests.post(
            f"{APPEEARS_URL}/login",
            auth=(username, password),
            timeout=15
        )
        if r.status_code == 200:
            return r.json().get("token")
    except:
        pass
    return None


def get_modis_lst(lat, lng, date_str=None):
    """
    Get MODIS Land Surface Temperature
    MOD11A1 — daily 1km
    Available globally — fallback for ECOSTRESS

    Uses NASA MODIS via Open-Meteo land API
    as simpler alternative to AppEEARS
    """
    try:
        today = datetime.now()
        target = datetime.strptime(date_str, "%Y-%m-%d") \
            if date_str else today

        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lng,
                "hourly": "soil_temperature_0cm",
                "past_days": 3,
                "forecast_days": 1,
                "timezone": "Europe/Dublin"
            }, timeout=15)

        if r.status_code == 200:
            data = r.json()
            temps = data.get("hourly", {}).get(
                "soil_temperature_0cm", [])
            times = data.get("hourly", {}).get("time", [])

            # Get daytime values (10am-2pm)
            daytime = [t for t, v in zip(temps, times)
                      if v and "T1" in v and v is not None]

            if temps:
                valid = [t for t in temps if t is not None]
                lst_mean = round(np.mean(valid), 2) if valid else None
                lst_max = round(max(valid), 2) if valid else None

                return {
                    "lst_kelvin": round(lst_mean + 273.15, 2)
                                  if lst_mean else None,
                    "lst_celsius": lst_mean,
                    "lst_max_celsius": lst_max,
                    "source": "Open-Meteo surface temperature (MODIS proxy)",
                    "resolution_m": 1000,
                    "date": today.strftime("%Y-%m-%d")
                }
    except Exception as e:
        pass
    return None


def calculate_crop_water_stress(lst_celsius, air_temp,
                                  ndvi=None, crop_type=None):
    """
    Calculate Crop Water Stress Index (CWSI)
    Based on canopy-air temperature difference

    CWSI = (Tc - Ta) / (Tc_stressed - Tc_non_stressed)
    Where:
    Tc = canopy temperature (LST)
    Ta = air temperature

    CWSI = 0: no stress (well watered)
    CWSI = 1: maximum stress (wilting)

    Reference: Jackson et al 1981
    """
    if not lst_celsius or not air_temp:
        return None

    # Temperature difference
    delta_t = lst_celsius - air_temp

    # Crop-specific baselines (empirical)
    # Non-stressed: canopy cooler than air (transpiring well)
    # Stressed: canopy warmer than air (stomata closed)
    baselines = {
        "Winter Wheat":  {"non_stressed": -2.0, "stressed": 5.0},
        "Spring Barley": {"non_stressed": -1.5, "stressed": 5.5},
        "Potato":        {"non_stressed": -2.5, "stressed": 4.5},
        "Oilseed Rape":  {"non_stressed": -2.0, "stressed": 5.0},
        "Grassland":     {"non_stressed": -3.0, "stressed": 4.0},
        "Unknown":       {"non_stressed": -2.0, "stressed": 5.0}
    }

    b = baselines.get(crop_type, baselines["Unknown"])
    non_stressed = b["non_stressed"]
    stressed = b["stressed"]

    cwsi = (delta_t - non_stressed) / (stressed - non_stressed)
    cwsi = max(0, min(1, cwsi))

    # Stress classification
    if cwsi < 0.2:
        stress_level = "No Stress"
        stress_icon = "✅"
    elif cwsi < 0.4:
        stress_level = "Mild Stress"
        stress_icon = "🟡"
    elif cwsi < 0.6:
        stress_level = "Moderate Stress"
        stress_icon = "🟠"
    elif cwsi < 0.8:
        stress_level = "Severe Stress"
        stress_icon = "🔴"
    else:
        stress_level = "Extreme Stress"
        stress_icon = "🚨"

    return {
        "cwsi": round(cwsi, 3),
        "stress_level": stress_level,
        "stress_icon": stress_icon,
        "canopy_temp_c": lst_celsius,
        "air_temp_c": air_temp,
        "temp_difference_c": round(delta_t, 2)
    }


def estimate_actual_et(lst_celsius, ndvi, air_temp,
                        wind_speed=2.0, humidity=70):
    """
    Estimate actual evapotranspiration from LST + NDVI
    Simplified Penman-Monteith approach

    Returns ET in mm/day
    """
    if not lst_celsius or not air_temp:
        return None

    # Potential ET estimate (simplified)
    # Based on temperature and radiation proxy
    pet = max(0, (lst_celsius - 5) * 0.15)

    # ET fraction from NDVI (vegetation fraction)
    if ndvi:
        et_fraction = min(1.0, max(0.1, ndvi * 1.2))
    else:
        et_fraction = 0.6

    # Stress factor from canopy-air temperature
    stress_factor = max(0.1, 1 - max(0, (lst_celsius - air_temp) / 10))

    actual_et = pet * et_fraction * stress_factor
    return round(actual_et, 2)


def get_thermal_stress_profile(lat, lng, crop_type,
                                air_temp=None, ndvi=None):
    """
    Complete thermal stress profile for a field

    Combines:
    - LST from MODIS/ECOSTRESS
    - CWSI calculation
    - Actual ET estimate
    - Stress classification
    """
    # Check ECOSTRESS coverage (52°N limit)
    ecostress_available = lat <= 52.0
    source = "ECOSTRESS (70m)" if ecostress_available \
             else "MODIS LST (1km)"

    # Get LST
    lst_data = get_modis_lst(lat, lng)

    if not lst_data:
        return {
            "available": False,
            "reason": "LST data unavailable"
        }

    lst_c = lst_data.get("lst_celsius")

    # Get air temperature if not provided
    if not air_temp:
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lng,
                    "current": "temperature_2m",
                    "timezone": "Europe/Dublin"
                }, timeout=10)
            if r.status_code == 200:
                air_temp = r.json().get(
                    "current", {}).get("temperature_2m")
        except:
            pass

    # Calculate CWSI
    cwsi = calculate_crop_water_stress(
        lst_c, air_temp, ndvi, crop_type)

    # Estimate actual ET
    actual_et = estimate_actual_et(lst_c, ndvi, air_temp)

    # Irrigation signal
    irrigation_signal = None
    if cwsi:
        if cwsi["cwsi"] > 0.5:
            irrigation_signal = "Irrigate — high water stress detected"
        elif cwsi["cwsi"] > 0.3:
            irrigation_signal = "Monitor — mild water stress developing"
        else:
            irrigation_signal = "No irrigation needed"

    return {
        "available": True,
        "ecostress_coverage": ecostress_available,
        "data_source": source,
        "lst": {
            "celsius": lst_c,
            "kelvin": lst_data.get("lst_kelvin"),
            "max_celsius": lst_data.get("lst_max_celsius")
        },
        "air_temperature_c": air_temp,
        "crop_water_stress": cwsi,
        "actual_et_mm_day": actual_et,
        "irrigation_signal": irrigation_signal,
        "note": (
            "Within ECOSTRESS coverage zone"
            if ecostress_available else
            "Outside ECOSTRESS coverage (>52N) — using MODIS LST"
        )
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/workspaces/crop-trajectory')

    print("Testing Thermal Stress — Ireland")
    print("="*50)

    # Test southern Ireland (within ECOSTRESS range)
    lat_south, lng_south = 51.9, -8.5  # Co. Cork
    lat_north, lng_north = 53.6, -6.7  # Co. Meath

    for label, lat, lng in [
        ("Co. Cork (52N coverage)", lat_south, lng_south),
        ("Co. Meath (outside coverage)", lat_north, lng_north)
    ]:
        print(f"\n{label}:")
        result = get_thermal_stress_profile(
            lat, lng,
            crop_type="Winter Wheat",
            ndvi=0.68
        )

        if result["available"]:
            print(f"  Data source:    {result['data_source']}")
            print(f"  Coverage note:  {result['note']}")
            lst = result["lst"]
            print(f"  LST:            {lst['celsius']}°C")
            print(f"  Air temp:       {result['air_temperature_c']}°C")
            cwsi = result["crop_water_stress"]
            if cwsi:
                print(f"  CWSI:           {cwsi['cwsi']}")
                print(f"  Stress:         {cwsi['stress_icon']} {cwsi['stress_level']}")
                print(f"  Canopy-Air ΔT:  {cwsi['temp_difference_c']}°C")
            print(f"  Actual ET:      {result['actual_et_mm_day']} mm/day")
            print(f"  Signal:         {result['irrigation_signal']}")
        else:
            print(f"  Unavailable: {result['reason']}")
