"""
Rice (Paddy) Growth Stage Model
Detects phenological stages from Sentinel-1 SAR time series

Based on Japanese JAXA/NARO research:
- VH backscatter increases during tillering
- VV/VH ratio changes at heading
- Sharp backscatter rise at heading when
  panicles bend from vertical to horizontal
- Signal decline during ripening

Indian Kharif Rice calendar (June-November):
Transplanting: June-July
Tillering: July-August  
Heading: September
Grain Fill: October
Harvest: November

Indian Rabi Rice (some regions):
Transplanting: November-December
Harvest: March-April
"""

import numpy as np
from datetime import datetime, timedelta


RICE_STAGES = {
    0:  "Land Preparation / Dry Field",
    10: "Transplanting / Direct Seeding",
    20: "Early Tillering",
    30: "Active Tillering",
    40: "Panicle Initiation",
    50: "Heading / Flowering",
    70: "Grain Fill / Milky Stage",
    87: "Ripening / Dough Stage",
    99: "Harvest"
}

# Kharif season (main Indian rice season)
# Days from transplanting (June/July assumed)
INDIA_RICE_TIMELINE = {
    0:  0,   # Transplanting
    10: 7,   # Early establishment
    20: 21,  # Early tillering
    30: 40,  # Active tillering
    40: 60,  # Panicle initiation
    50: 75,  # Heading
    70: 90,  # Grain fill
    87: 110, # Ripening
    99: 125  # Harvest
}


def calculate_vh_vv_ratio(vv_series, vh_series):
    ratios = []
    for vv, vh in zip(vv_series, vh_series):
        if vv and vh and vv > 0:
            ratios.append(round(vh / vv, 4))
        else:
            ratios.append(None)
    return ratios


def detect_transplanting(vv_series, vh_series, dates):
    """
    Detect transplanting date
    Signature: Sharp VV drop when field flooded
    Then VH rise as seedlings establish
    Kharif: June-July
    """
    for i in range(1, len(vv_series)):
        if dates[i].month not in [6, 7, 11, 12]:
            continue
        if vv_series[i] and vv_series[i-1]:
            drop = vv_series[i] / vv_series[i-1]
            if drop < 0.80:  # Field flooding signal
                return dates[i], dates[i] + timedelta(days=7)
    return None, None


def detect_heading_rice(vv_series, vh_series, dates, transplanting=None):
    """
    Detect rice heading from SAR
    Signature: Sharp backscatter rise when
    panicles bend from vertical to horizontal
    Same physics as barley ear emergence
    Kharif heading: September-October
    Minimum 60 days after transplanting
    """
    for i in range(1, len(vv_series)):
        if dates[i].month not in [8, 9, 10, 2, 3]:
            continue
        if transplanting:
            if (dates[i] - transplanting).days < 55:
                continue
        if vv_series[i] and vv_series[i-1]:
            rise = vv_series[i] / vv_series[i-1]
            if rise > 1.30:
                return dates[i], rise
    return None, None


def detect_harvest_rice(vv_series, dates, transplanting=None):
    """
    Detect rice harvest
    Signature: Sharp drop — field drained + crop cut
    Kharif harvest: October-November
    """
    for i in range(1, len(vv_series)):
        if dates[i].month not in [10, 11, 12, 3, 4]:
            continue
        if transplanting:
            if (dates[i] - transplanting).days < 100:
                continue
        if vv_series[i] and vv_series[i-1]:
            drop = vv_series[i] / vv_series[i-1]
            if drop < 0.75:
                return dates[i]
    return None


def estimate_rice_stage(observations, season="kharif"):
    """
    Main function — estimates current rice growth stage
    """
    if not observations:
        return {"error": "No observations provided"}

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
    transplanting, establishment = detect_transplanting(
        vv_series, vh_series, dates)
    heading, heading_conf = detect_heading_rice(
        vv_series, vh_series, dates, transplanting)
    harvest = detect_harvest_rice(vv_series, dates, transplanting)

    latest = dates[-1]
    days_since_transplanting = (
        latest - transplanting).days if transplanting else None

    # Current stage from days
    current_bbch = 0
    current_stage = RICE_STAGES[0]

    if transplanting and days_since_transplanting:
        for bbch, days in sorted(
                INDIA_RICE_TIMELINE.items(), reverse=True):
            if days_since_transplanting >= days:
                current_bbch = bbch
                current_stage = RICE_STAGES[bbch]
                break

    # Override with detected events
    if harvest and latest >= harvest:
        current_bbch = 99
        current_stage = RICE_STAGES[99]
    elif heading and latest >= heading:
        if 50 > current_bbch:
            current_bbch = 50
            current_stage = RICE_STAGES[50]

    # Next stage
    stage_list = sorted(INDIA_RICE_TIMELINE.keys())
    current_idx = stage_list.index(current_bbch) \
        if current_bbch in stage_list else 0
    next_bbch = None
    next_stage = None
    days_to_next = None

    if current_idx < len(stage_list) - 1:
        next_bbch = stage_list[current_idx + 1]
        next_stage = RICE_STAGES[next_bbch]
        if days_since_transplanting is not None:
            days_to_next = max(
                0, INDIA_RICE_TIMELINE[next_bbch] - days_since_transplanting)

    # Management alerts
    alerts = []
    if current_bbch == 30:
        alerts.append("Active tillering — top dress nitrogen now")
        alerts.append("Monitor for brown planthopper")
    if current_bbch == 40:
        alerts.append("Panicle initiation — critical water management stage")
        alerts.append("Maintain flooding depth")
    if current_bbch == 50:
        alerts.append("Heading — apply potassium if not done")
        alerts.append("Monitor for blast disease")
    if current_bbch == 87:
        alerts.append("Ripening — prepare harvest equipment")
        alerts.append("Drain field 10-14 days before harvest")

    return {
        "crop": "Rice (Paddy)",
        "season": season,
        "location": "India",
        "latest_observation": latest.strftime("%Y-%m-%d"),
        "current_bbch": current_bbch,
        "current_stage": current_stage,
        "next_bbch": next_bbch,
        "next_stage": next_stage,
        "days_to_next_stage": days_to_next,
        "transplanting_detected": transplanting.strftime("%Y-%m-%d")
            if transplanting else None,
        "heading_detected": heading.strftime("%Y-%m-%d")
            if heading else None,
        "harvest_detected": harvest.strftime("%Y-%m-%d")
            if harvest else None,
        "days_since_transplanting": days_since_transplanting,
        "management_alerts": alerts,
        "confidence": "HIGH" if heading else
                      "MEDIUM" if transplanting else "LOW"
    }


if __name__ == "__main__":
    import json, os, sys
    sys.path.insert(0, '/workspaces/crop-trajectory')

    os.environ['CDSE_CLIENT_ID'] = 'sh-6e5978f5-f5d6-43d6-874d-720d84121683'
    os.environ['CDSE_CLIENT_SECRET'] = 'yrMEXQ5drlF26yrB4sTEXfWOIwKtB1fP'

    from extractors.sar_timeseries import get_sar_timeseries

    # Test on known Paddy location from dataset
    # Andhra Pradesh rice growing area
    lat, lng = 15.9128998, 79.7399875

    print(f"Testing Rice Model — Andhra Pradesh ({lat}, {lng})")
    print("="*55)

    # Kharif season 2025
    obs = get_sar_timeseries(
        lat, lng,
        '2025-06-01', '2025-12-01',
        os.environ['CDSE_CLIENT_ID'],
        os.environ['CDSE_CLIENT_SECRET'],
        interval_days=6
    )
    available = [o for o in obs if o.get('available')]
    print(f"Got {len(available)} observations")

    if available:
        result = estimate_rice_stage(available)
        print(f"\nCrop:              {result['crop']}")
        print(f"Season:            {result['season']}")
        print(f"Current stage:     {result['current_bbch']} — {result['current_stage']}")
        print(f"Next stage:        {result['next_bbch']} — {result['next_stage']}")
        print(f"Days to next:      {result['days_to_next_stage']}")
        print(f"Transplanting:     {result['transplanting_detected']}")
        print(f"Heading:           {result['heading_detected']}")
        print(f"Confidence:        {result['confidence']}")
        if result['management_alerts']:
            print("Alerts:")
            for a in result['management_alerts']:
                print(f"  → {a}")
