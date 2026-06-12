"""
Spring Barley Growth Stage Model
Detects BBCH growth stages from Sentinel-1 SAR time series
Based on VH/VV ratio breakpoint detection

Key signatures:
- Sowing: low VH/VV ratio, stable
- Emergence: VH starts rising
- Tillering: both VH and VV rising
- Stem Extension: VH peak building
- Heading: sharp 8dB rise in backscatter
- Ripening: signal declining
- Harvest: sharp drop to baseline
"""

import numpy as np
from datetime import datetime, timedelta


# BBCH stage definitions for Spring Barley
BBCH_STAGES = {
    0:  "Sowing / Bare Soil",
    10: "Emergence",
    20: "Tillering",
    30: "Stem Extension",
    49: "Booting",
    55: "Heading",
    73: "Grain Fill",
    87: "Ripening",
    99: "Harvest / Post-Harvest"
}

# Typical Spring Barley calendar for Ireland
# Days from sowing (April sowing assumed)
IRELAND_BARLEY_TIMELINE = {
    0:  0,   # Sowing — Day 0
    10: 14,  # Emergence — Day 14
    20: 35,  # Tillering — Day 35
    30: 65,  # Stem Extension — Day 65
    49: 80,  # Booting — Day 80
    55: 90,  # Heading — Day 90
    73: 105, # Grain Fill — Day 105
    87: 120, # Ripening — Day 120
    99: 140  # Harvest — Day 140
}


def calculate_ratio(vv_series, vh_series):
    """Calculate VH/VV ratio time series"""
    ratios = []
    for vv, vh in zip(vv_series, vh_series):
        if vv and vh and vv > 0:
            ratios.append(vh / vv)
        else:
            ratios.append(None)
    return ratios


def calculate_velocity(values, dates):
    """Calculate rate of change between observations"""
    velocities = [None]
    for i in range(1, len(values)):
        if values[i] and values[i-1]:
            days = (dates[i] - dates[i-1]).days
            if days > 0:
                vel = (values[i] - values[i-1]) / days
                velocities.append(round(vel, 6))
            else:
                velocities.append(None)
        else:
            velocities.append(None)
    return velocities


def detect_sowing(vv_series, vh_series, dates):
    """
    Detect sowing date from SAR
    Signature: Low VH/VV ratio period followed by VH rise
    """
    ratios = calculate_ratio(vv_series, vh_series)
    
    for i in range(1, len(ratios)):
        if ratios[i] and ratios[i-1]:
            # VH rising from low baseline = emergence after sowing
            if ratios[i] > ratios[i-1] * 1.05:
                # Sowing approximately 14 days before emergence
                sowing_date = dates[i] - timedelta(days=14)
                return sowing_date, dates[i]
    
    return None, None


def detect_heading(vv_series, vh_series, dates, sowing_date=None):
    """
    Detect heading date from SAR
    Signature: Sharp 8dB rise in backscatter
    Barley ears bend from vertical to horizontal
    This is the most reliable detection point
    
    For Irish Spring Barley:
    Heading occurs 85-100 days after sowing
    Minimum 70 days after sowing to avoid false positives
    """
    for i in range(1, len(vv_series)):
        if vv_series[i] and vv_series[i-1]:
            
            # Only look for heading after minimum days from sowing
            if sowing_date:
                days_from_sowing = (dates[i] - sowing_date).days
                if days_from_sowing < 70:
                    continue
            else:
                # Without sowing date, only look from June onwards for Ireland
                if dates[i].month < 6:
                    continue
            
            # Check for sharp rise in VV
            # Irish barley heading spike typically 1.3-1.8x
            rise_ratio = vv_series[i] / vv_series[i-1] if vv_series[i-1] > 0 else 0
            if rise_ratio > 1.3:
                return dates[i], rise_ratio
    
    return None, None


def detect_harvest(vv_series, vh_series, dates, sowing_date=None):
    """
    Detect harvest date from SAR
    Signature: Sharp drop back to bare soil values
    
    For Irish Spring Barley:
    Harvest occurs 120-150 days after sowing
    Minimum 110 days after sowing to avoid false positives
    """
    for i in range(1, len(vv_series)):
        if vv_series[i] and vv_series[i-1]:

            # Only look for harvest after minimum days from sowing
            if sowing_date:
                days_from_sowing = (dates[i] - sowing_date).days
                if days_from_sowing < 110:
                    continue
            else:
                # Without sowing date only look from July onwards
                if dates[i].month < 7:
                    continue

            drop_ratio = vv_series[i] / vv_series[i-1] if vv_series[i-1] > 0 else 0
            if drop_ratio < 0.75:  # Sharp decline
                return dates[i]

    return None


def estimate_current_stage(observations, crop_year=None):
    """
    Main function — estimates current BBCH growth stage
    from SAR time series observations
    
    Args:
        observations: list of dicts with date, vv, vh, rvi
        crop_year: year of crop (optional)
    
    Returns:
        dict with current stage, next stage, days to next stage
    """
    if not observations:
        return {"error": "No observations provided"}
    
    # Extract series
    dates = []
    vv_series = []
    vh_series = []
    rvi_series = []
    
    for obs in observations:
        if obs.get("available") and obs.get("vv") and obs.get("vh"):
            dates.append(datetime.strptime(obs["date"], "%Y-%m-%d"))
            vv_series.append(obs["vv"])
            vh_series.append(obs["vh"])
            rvi_series.append(obs.get("rvi", 0))
    
    if len(dates) < 3:
        return {"error": "Insufficient observations for stage detection"}
    
    # Calculate ratios and velocities
    ratios = calculate_ratio(vv_series, vh_series)
    velocities = calculate_velocity(rvi_series, dates)
    
    # Detect key events
    sowing_date, emergence_date = detect_sowing(vv_series, vh_series, dates)
    heading_date, heading_confidence = detect_heading(vv_series, vh_series, dates, sowing_date)
    harvest_date = detect_harvest(vv_series, vh_series, dates, sowing_date)
    
    # Current date context
    latest_date = dates[-1]
    latest_vv = vv_series[-1]
    latest_vh = vh_series[-1]
    latest_rvi = rvi_series[-1] if rvi_series else 0
    latest_ratio = ratios[-1] if ratios else 0
    
    # Determine current stage from detected events
    current_bbch = 0
    current_stage_name = BBCH_STAGES[0]
    days_since_sowing = None
    
    if sowing_date:
        days_since_sowing = (latest_date - sowing_date).days
        
        # Estimate stage from days since sowing
        for bbch, days in sorted(IRELAND_BARLEY_TIMELINE.items(), reverse=True):
            if days_since_sowing >= days:
                current_bbch = bbch
                current_stage_name = BBCH_STAGES[bbch]
                break
    
    # Override with detected heading if found
    if heading_date and latest_date >= heading_date:
        if harvest_date and latest_date >= harvest_date:
            current_bbch = 99
            current_stage_name = BBCH_STAGES[99]
        else:
            current_bbch = 55
            current_stage_name = BBCH_STAGES[55]
    
    # Calculate next stage
    next_bbch = None
    next_stage_name = None
    days_to_next = None
    
    stage_list = sorted(IRELAND_BARLEY_TIMELINE.keys())
    current_idx = stage_list.index(current_bbch) if current_bbch in stage_list else 0
    
    if current_idx < len(stage_list) - 1:
        next_bbch = stage_list[current_idx + 1]
        next_stage_name = BBCH_STAGES[next_bbch]
        if sowing_date:
            days_to_next_from_sowing = IRELAND_BARLEY_TIMELINE[next_bbch]
            days_elapsed = days_since_sowing or 0
            days_to_next = max(0, days_to_next_from_sowing - days_elapsed)
    
    return {
        "crop": "Spring Barley",
        "location": "Ireland",
        "latest_observation": latest_date.strftime("%Y-%m-%d"),
        "current_bbch": current_bbch,
        "current_stage": current_stage_name,
        "next_bbch": next_bbch,
        "next_stage": next_stage_name,
        "days_to_next_stage": days_to_next,
        "sowing_date_detected": sowing_date.strftime("%Y-%m-%d") if sowing_date else None,
        "heading_date_detected": heading_date.strftime("%Y-%m-%d") if heading_date else None,
        "harvest_date_detected": harvest_date.strftime("%Y-%m-%d") if harvest_date else None,
        "days_since_sowing": days_since_sowing,
        "latest_sar": {
            "vv": latest_vv,
            "vh": latest_vh,
            "rvi": latest_rvi,
            "vh_vv_ratio": round(latest_ratio, 4) if latest_ratio else None
        },
        "confidence": "HIGH" if heading_date else "MEDIUM" if sowing_date else "LOW"
    }


if __name__ == "__main__":
    import json
    import os
    import sys
    sys.path.insert(0, '/workspaces/crop-trajectory')
    
    # Test with existing SAR data
    # Use an Irish field coordinate
    print("Testing Spring Barley Growth Model")
    print("="*50)
    
    # Load existing SAR data if available
    test_file = "/workspaces/crop-trajectory/sar_timeseries_test.json"
    
    if os.path.exists(test_file):
        with open(test_file) as f:
            observations = json.load(f)
        
        result = estimate_current_stage(observations)
        
        print(f"Crop: {result['crop']}")
        print(f"Latest observation: {result['latest_observation']}")
        print(f"Current stage: BBCH {result['current_bbch']} — {result['current_stage']}")
        print(f"Next stage: BBCH {result['next_bbch']} — {result['next_stage']}")
        print(f"Days to next stage: {result['days_to_next_stage']}")
        print(f"Sowing detected: {result['sowing_date_detected']}")
        print(f"Heading detected: {result['heading_date_detected']}")
        print(f"Harvest detected: {result['harvest_date_detected']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Latest SAR: {result['latest_sar']}")
    else:
        print("No SAR data found. Run sar_timeseries.py first.")
