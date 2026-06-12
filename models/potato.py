"""
Potato Growth Stage Model
Detects phenological stages from Sentinel-1 SAR time series
Combined with NDVI where available

Key signatures:
- Planting: high VV (ridged soil), low VH
- Emergence: VH starts rising
- Canopy Closure: VH rising strongly, VH/VV peak
- Flowering: VH maximum
- Senescence: both declining
- Harvest: sharp drop to baseline

Reference: Sentinel-1 + Sentinel-2 potato phenology detection
RMSE 12-14 days for emergence and canopy closure
R² 0.80 for integrated NDVI_VH approach
"""

import numpy as np
from datetime import datetime, timedelta


# Potato growth stages for Ireland
POTATO_STAGES = {
    0:  "Planting / Bare Soil",
    10: "Emergence",
    30: "Canopy Development",
    60: "Canopy Closure",
    65: "Flowering / Tuber Initiation",
    80: "Tuber Bulking",
    90: "Senescence",
    99: "Harvest"
}

# Typical Irish Maincrop Potato calendar
# Days from planting (April/May planting assumed)
IRELAND_POTATO_TIMELINE = {
    0:  0,   # Planting — Day 0
    10: 21,  # Emergence — Day 21
    30: 45,  # Canopy Development — Day 45
    60: 65,  # Canopy Closure — Day 65
    65: 80,  # Flowering/Tuber Initiation — Day 80
    80: 100, # Tuber Bulking — Day 100
    90: 130, # Senescence — Day 130
    99: 155  # Harvest — Day 155
}


def calculate_vh_vv_ratio(vv_series, vh_series):
    """Calculate VH/VV ratio — key index for potato stages"""
    ratios = []
    for vv, vh in zip(vv_series, vh_series):
        if vv and vh and vv > 0:
            ratios.append(round(vh / vv, 4))
        else:
            ratios.append(None)
    return ratios


def smooth_series(values, window=3):
    """Simple moving average smoothing"""
    smoothed = []
    for i in range(len(values)):
        valid = [v for v in values[max(0,i-window):i+window+1] if v]
        smoothed.append(round(sum(valid)/len(valid), 4) if valid else None)
    return smoothed


def find_peak(values, dates, min_date=None, max_date=None):
    """Find peak value in series within date range"""
    peak_val = None
    peak_date = None
    for i, (val, date) in enumerate(zip(values, dates)):
        if val is None:
            continue
        if min_date and date < min_date:
            continue
        if max_date and date > max_date:
            continue
        if peak_val is None or val > peak_val:
            peak_val = val
            peak_date = date
    return peak_date, peak_val


def detect_planting(vv_series, vh_series, dates):
    """
    Detect planting date from SAR
    Signature: High VV (ridged soil) followed by VH rise
    Irish potatoes planted April-May
    """
    ratios = calculate_vh_vv_ratio(vv_series, vh_series)

    for i in range(1, len(ratios)):
        if ratios[i] and ratios[i-1]:
            # Look for start of VH rise after low ratio period
            if ratios[i] > ratios[i-1] * 1.08:
                # Only in April-June
                if dates[i].month in [4, 5, 6]:
                    planting_date = dates[i] - timedelta(days=21)
                    emergence_date = dates[i]
                    return planting_date, emergence_date

    return None, None


def detect_canopy_closure(vv_series, vh_series, dates, planting_date=None):
    """
    Detect canopy closure from SAR
    Signature: VH/VV ratio reaches peak
    Occurs approximately 65 days after planting
    """
    ratios = calculate_vh_vv_ratio(vv_series, vh_series)
    smoothed = smooth_series(ratios)

    # Look for peak in VH/VV ratio
    # Only from June onwards and min 50 days after planting
    min_date = None
    if planting_date:
        min_date = planting_date + timedelta(days=50)
    else:
        # Default to June
        year = dates[0].year if dates else datetime.now().year
        min_date = datetime(year, 6, 1)

    max_date = None
    if planting_date:
        max_date = planting_date + timedelta(days=100)

    peak_date, peak_val = find_peak(smoothed, dates, min_date, max_date)
    return peak_date, peak_val


def detect_senescence(vv_series, vh_series, dates, planting_date=None):
    """
    Detect senescence onset from SAR
    Signature: VH starts declining after peak
    Occurs approximately 130 days after planting
    """
    ratios = calculate_vh_vv_ratio(vv_series, vh_series)
    smoothed = smooth_series(ratios)

    # Find peak first
    min_date = planting_date + timedelta(days=70) if planting_date else None
    peak_date, _ = find_peak(smoothed, dates, min_date)

    if not peak_date:
        return None

    # Look for consistent decline after peak
    peak_idx = dates.index(peak_date) if peak_date in dates else None
    if peak_idx is None:
        return None

    for i in range(peak_idx + 1, len(smoothed)):
        if smoothed[i] and smoothed[peak_idx]:
            decline = (smoothed[peak_idx] - smoothed[i]) / smoothed[peak_idx]
            if decline > 0.15:  # 15% decline from peak
                return dates[i]

    return None


def detect_harvest_potato(vv_series, dates, planting_date=None):
    """
    Detect harvest from SAR
    Signature: Sharp drop in VV backscatter
    Occurs approximately 155 days after planting
    """
    min_days = 120
    for i in range(1, len(vv_series)):
        if vv_series[i] and vv_series[i-1]:
            if planting_date:
                days_from_planting = (dates[i] - planting_date).days
                if days_from_planting < min_days:
                    continue
            else:
                if dates[i].month < 8:
                    continue

            drop_ratio = vv_series[i] / vv_series[i-1] if vv_series[i-1] > 0 else 1
            if drop_ratio < 0.75:
                return dates[i]

    return None


def estimate_potato_stage(observations):
    """
    Main function — estimates current potato growth stage
    from SAR time series

    Args:
        observations: list of dicts with date, vv, vh, rvi

    Returns:
        dict with current stage, next stage, management alerts
    """
    if not observations:
        return {"error": "No observations provided"}

    # Extract series
    dates = []
    vv_series = []
    vh_series = []

    for obs in observations:
        if obs.get("available") and obs.get("vv") and obs.get("vh"):
            dates.append(datetime.strptime(obs["date"], "%Y-%m-%d"))
            vv_series.append(obs["vv"])
            vh_series.append(obs["vh"])

    if len(dates) < 3:
        return {"error": "Insufficient observations"}

    # Detect key events
    planting_date, emergence_date = detect_planting(vv_series, vh_series, dates)
    canopy_closure_date, _ = detect_canopy_closure(
        vv_series, vh_series, dates, planting_date)
    senescence_date = detect_senescence(
        vv_series, vh_series, dates, planting_date)
    harvest_date = detect_harvest_potato(vv_series, dates, planting_date)

    # Current context
    latest_date = dates[-1]
    days_since_planting = (latest_date - planting_date).days if planting_date else None

    # Determine current stage
    current_stage_id = 0
    current_stage_name = POTATO_STAGES[0]

    if planting_date and days_since_planting:
        for stage_id, days in sorted(
                IRELAND_POTATO_TIMELINE.items(), reverse=True):
            if days_since_planting >= days:
                current_stage_id = stage_id
                current_stage_name = POTATO_STAGES[stage_id]
                break

    # Override with detected events only if they are more advanced
    if harvest_date and latest_date >= harvest_date:
        if 99 > current_stage_id:
            current_stage_id = 99
            current_stage_name = POTATO_STAGES[99]
    elif senescence_date and latest_date >= senescence_date:
        if 90 > current_stage_id:
            current_stage_id = 90
            current_stage_name = POTATO_STAGES[90]
    elif canopy_closure_date and latest_date >= canopy_closure_date:
        if 65 > current_stage_id:
            current_stage_id = 65
            current_stage_name = POTATO_STAGES[65]

    # Next stage
    stage_list = sorted(IRELAND_POTATO_TIMELINE.keys())
    current_idx = stage_list.index(current_stage_id) \
        if current_stage_id in stage_list else 0
    next_stage_id = None
    next_stage_name = None
    days_to_next = None

    if current_idx < len(stage_list) - 1:
        next_stage_id = stage_list[current_idx + 1]
        next_stage_name = POTATO_STAGES[next_stage_id]
        if planting_date and days_since_planting:
            days_to_next_from_planting = IRELAND_POTATO_TIMELINE[next_stage_id]
            days_to_next = max(0, days_to_next_from_planting - days_since_planting)

    # Management alerts
    alerts = []
    if current_stage_id == 65:
        alerts.append("Tuber initiation — calcium application window open")
        alerts.append("Monitor soil moisture — critical for tuber set")
    if current_stage_id == 80:
        alerts.append("Tuber bulking — maintain soil moisture")
        alerts.append("Potassium demand at maximum")
    if current_stage_id == 90:
        alerts.append("Senescence — prepare for harvest scheduling")
        alerts.append("Desiccation timing window approaching")

    confidence = "HIGH" if canopy_closure_date else \
                 "MEDIUM" if planting_date else "LOW"

    return {
        "crop": "Potato",
        "variety_type": "Maincrop",
        "location": "Ireland",
        "latest_observation": latest_date.strftime("%Y-%m-%d"),
        "current_stage_id": current_stage_id,
        "current_stage": current_stage_name,
        "next_stage_id": next_stage_id,
        "next_stage": next_stage_name,
        "days_to_next_stage": days_to_next,
        "planting_date_detected": planting_date.strftime("%Y-%m-%d") \
            if planting_date else None,
        "emergence_date_detected": emergence_date.strftime("%Y-%m-%d") \
            if emergence_date else None,
        "canopy_closure_detected": canopy_closure_date.strftime("%Y-%m-%d") \
            if canopy_closure_date else None,
        "senescence_detected": senescence_date.strftime("%Y-%m-%d") \
            if senescence_date else None,
        "harvest_detected": harvest_date.strftime("%Y-%m-%d") \
            if harvest_date else None,
        "days_since_planting": days_since_planting,
        "management_alerts": alerts,
        "confidence": confidence
    }


if __name__ == "__main__":
    import json, os, sys
    sys.path.insert(0, '/workspaces/crop-trajectory')

    print("Testing Potato Growth Model — Ireland")
    print("="*50)

    # Use Irish barley data as proxy for now
    # Will replace with actual potato field coordinates
    test_file = "/workspaces/crop-trajectory/sar_ireland_barley.json"

    if os.path.exists(test_file):
        with open(test_file) as f:
            observations = json.load(f)

        result = estimate_potato_stage(observations)

        print(f"Crop:              {result['crop']}")
        print(f"Latest obs:        {result['latest_observation']}")
        print(f"Current stage:     {result['current_stage_id']} — {result['current_stage']}")
        print(f"Next stage:        {result['next_stage_id']} — {result['next_stage']}")
        print(f"Days to next:      {result['days_to_next_stage']}")
        print(f"Planting detected: {result['planting_date_detected']}")
        print(f"Canopy closure:    {result['canopy_closure_detected']}")
        print(f"Senescence:        {result['senescence_detected']}")
        print(f"Harvest:           {result['harvest_detected']}")
        print(f"Days since plant:  {result['days_since_planting']}")
        print(f"Confidence:        {result['confidence']}")
        if result['management_alerts']:
            print(f"Alerts:")
            for alert in result['management_alerts']:
                print(f"  → {alert}")
