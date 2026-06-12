"""
Crop Physiological Parameters
Estimates LAI, Biomass and canopy parameters
from Sentinel-1 SAR time series

Based on established relationships:
- VH backscatter correlates with LAI
- VH/VV ratio correlates with canopy closure
- Biomass estimated from dual-pol SAR

References:
- Hosseini et al 2015 — LAI from SAR
- Harfenmeister et al 2019 — SAR crop parameters
- Origin Digital Crop AI — LAI/Biomass/Growth stages
"""

import numpy as np
from datetime import datetime


# Crop-specific calibration parameters
# LAI sensitivity coefficients (dB per LAI unit)
LAI_PARAMS = {
    "Winter Wheat": {
        "vh_sensitivity": 0.80,
        "baseline_vh": 8.0,
        "max_lai": 7.0,
        "peak_stage": "Heading"
    },
    "Spring Barley": {
        "vh_sensitivity": 0.75,
        "baseline_vh": 7.5,
        "max_lai": 6.0,
        "peak_stage": "Heading"
    },
    "Potato": {
        "vh_sensitivity": 0.60,
        "baseline_vh": 6.0,
        "max_lai": 5.0,
        "peak_stage": "Canopy Closure"
    },
    "Oilseed Rape": {
        "vh_sensitivity": 0.70,
        "baseline_vh": 7.0,
        "max_lai": 5.5,
        "peak_stage": "Flowering"
    },
    "Grassland": {
        "vh_sensitivity": 0.50,
        "baseline_vh": 5.0,
        "max_lai": 4.0,
        "peak_stage": "Peak Growth"
    }
}

# Biomass coefficients (linear model)
# Biomass (t/ha DM) = a*VH + b*VV + c
BIOMASS_PARAMS = {
    "Winter Wheat": {"a": 0.25, "b": 0.08, "c": -3.5},
    "Spring Barley": {"a": 0.22, "b": 0.07, "c": -3.0},
    "Potato": {"a": 0.35, "b": 0.10, "c": -4.0},
    "Oilseed Rape": {"a": 0.28, "b": 0.09, "c": -3.8},
    "Grassland": {"a": 0.18, "b": 0.06, "c": -2.0}
}


def estimate_lai(vh_value, crop_type, baseline_vh=None):
    """
    Estimate Leaf Area Index from VH backscatter
    
    LAI = (VH_current - VH_baseline) / sensitivity
    Clamped to 0 - max_lai range
    """
    params = LAI_PARAMS.get(crop_type, LAI_PARAMS["Winter Wheat"])
    baseline = baseline_vh or params["baseline_vh"]
    sensitivity = params["vh_sensitivity"]
    max_lai = params["max_lai"]

    if not vh_value or vh_value <= 0:
        return None

    lai = (vh_value - baseline) / sensitivity
    lai = max(0, min(max_lai, lai))
    return round(lai, 2)


def estimate_biomass(vv_value, vh_value, crop_type):
    """
    Estimate above-ground biomass from dual-pol SAR
    Returns biomass in t/ha dry matter
    """
    params = BIOMASS_PARAMS.get(crop_type, BIOMASS_PARAMS["Winter Wheat"])

    if not vv_value or not vh_value:
        return None

    biomass = params["a"] * vh_value + params["b"] * vv_value + params["c"]
    biomass = max(0, biomass)
    return round(biomass, 2)


def calculate_canopy_cover(vh_vv_ratio, crop_type):
    """
    Estimate canopy cover percentage from VH/VV ratio
    0% = bare soil, 100% = full canopy closure
    """
    if not vh_vv_ratio:
        return None

    # Crop-specific thresholds
    thresholds = {
        "Winter Wheat":  {"bare": 0.12, "full": 0.22},
        "Spring Barley": {"bare": 0.12, "full": 0.20},
        "Potato":        {"bare": 0.14, "full": 0.24},
        "Oilseed Rape":  {"bare": 0.13, "full": 0.23},
        "Grassland":     {"bare": 0.14, "full": 0.20}
    }

    t = thresholds.get(crop_type, thresholds["Winter Wheat"])
    bare = t["bare"]
    full = t["full"]

    cover = (vh_vv_ratio - bare) / (full - bare) * 100
    cover = max(0, min(100, cover))
    return round(cover, 1)


def get_physiological_timeseries(observations, crop_type):
    """
    Calculate LAI, Biomass, Canopy Cover
    for each observation in the time series
    
    Returns list of physiological observations
    """
    if not observations:
        return []

    # Get baseline VH from early season
    early_vh = []
    for obs in observations[:5]:
        if obs.get("available") and obs.get("vh"):
            early_vh.append(obs["vh"])
    baseline_vh = sum(early_vh) / len(early_vh) if early_vh else None

    results = []
    for obs in observations:
        if not obs.get("available"):
            continue

        vv = obs.get("vv")
        vh = obs.get("vh")
        date = obs.get("date")

        if not vv or not vh:
            continue

        vh_vv = round(vh / vv, 4) if vv > 0 else None
        lai = estimate_lai(vh, crop_type, baseline_vh)
        biomass = estimate_biomass(vv, vh, crop_type)
        canopy = calculate_canopy_cover(vh_vv, crop_type)

        results.append({
            "date": date,
            "vv": vv,
            "vh": vh,
            "vh_vv_ratio": vh_vv,
            "lai": lai,
            "biomass_t_ha": biomass,
            "canopy_cover_pct": canopy
        })

    return results


def get_current_physiology(observations, crop_type):
    """
    Get current physiological status — latest observation
    """
    timeseries = get_physiological_timeseries(observations, crop_type)

    if not timeseries:
        return None

    latest = timeseries[-1]

    # Find peak LAI in season
    peak_lai = max([o["lai"] for o in timeseries if o["lai"]], default=0)
    peak_biomass = max(
        [o["biomass_t_ha"] for o in timeseries if o["biomass_t_ha"]],
        default=0)

    # LAI trend — use last 5 observations smoothed
    recent_lai = [o["lai"] for o in timeseries[-5:] if o["lai"]]
    if len(recent_lai) >= 3:
        first_half = sum(recent_lai[:len(recent_lai)//2]) / (len(recent_lai)//2)
        second_half = sum(recent_lai[len(recent_lai)//2:]) / len(recent_lai[len(recent_lai)//2:])
        diff = second_half - first_half
        if diff > 0.2:
            lai_trend = "increasing"
        elif diff < -0.2:
            lai_trend = "declining"
        else:
            lai_trend = "stable"
    elif len(recent_lai) > 1:
        lai_trend = "increasing" if recent_lai[-1] > recent_lai[0] else "declining"
    else:
        lai_trend = "stable"

    return {
        "current_lai": latest["lai"],
        "current_biomass_t_ha": latest["biomass_t_ha"],
        "current_canopy_cover_pct": latest["canopy_cover_pct"],
        "peak_lai_season": round(peak_lai, 2),
        "peak_biomass_t_ha": round(peak_biomass, 2),
        "lai_trend": lai_trend,
        "vh_vv_ratio": latest["vh_vv_ratio"],
        "date": latest["date"]
    }


if __name__ == "__main__":
    import json, sys
    sys.path.insert(0, '/workspaces/crop-trajectory')

    print("Testing Physiological Parameters — Ireland 2026")
    print("="*55)

    with open("/workspaces/crop-trajectory/sar_ireland_2026.json") as f:
        observations = json.load(f)

    available = [o for o in observations if o.get("available")]

    for crop in ["Winter Wheat", "Spring Barley", "Potato", "Oilseed Rape"]:
        print(f"\n{crop}:")
        result = get_current_physiology(available, crop)
        if result:
            print(f"  LAI:          {result['current_lai']}")
            print(f"  Biomass:      {result['current_biomass_t_ha']} t/ha DM")
            print(f"  Canopy cover: {result['current_canopy_cover_pct']}%")
            print(f"  Peak LAI:     {result['peak_lai_season']}")
            print(f"  LAI trend:    {result['lai_trend']}")
            print(f"  Peak biomass: {result['peak_biomass_t_ha']} t/ha")
