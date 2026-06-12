"""
Crop Type Classifier
Identifies crop type from Sentinel-1 SAR time series
Based on seasonal backscatter patterns and key event timing

Crops supported:
- Winter Wheat
- Spring Barley  
- Potato
- Oilseed Rape
- Grassland (permanent)
- Unknown/Other
"""

import numpy as np
from datetime import datetime
import sys
sys.path.insert(0, '/workspaces/crop-trajectory')
from models.spring_barley import estimate_current_stage
from models.winter_wheat import estimate_wheat_stage
from models.potato import estimate_potato_stage
from models.oilseed_rape import estimate_osr_stage


def get_seasonal_stats(observations):
    """
    Extract seasonal statistics from SAR time series
    Used as features for crop classification
    """
    dates = []
    vv_series = []
    vh_series = []
    ratios = []

    for obs in observations:
        if obs.get("available") and obs.get("vv") and obs.get("vh"):
            d = datetime.strptime(obs["date"], "%Y-%m-%d")
            dates.append(d)
            vv_series.append(obs["vv"])
            vh_series.append(obs["vh"])
            ratios.append(obs["vh"] / obs["vv"] if obs["vv"] > 0 else 0)

    if not dates:
        return None

    # Monthly averages
    monthly_vv = {}
    monthly_vh = {}
    for d, vv, vh in zip(dates, vv_series, vh_series):
        m = d.month
        if m not in monthly_vv:
            monthly_vv[m] = []
            monthly_vh[m] = []
        monthly_vv[m].append(vv)
        monthly_vh[m].append(vh)

    avg_vv = {m: round(sum(v)/len(v), 2) for m, v in monthly_vv.items()}
    avg_vh = {m: round(sum(v)/len(v), 2) for m, v in monthly_vh.items()}

    # Key statistics
    overall_mean_vv = round(np.mean(vv_series), 2)
    overall_mean_vh = round(np.mean(vh_series), 2)
    overall_std_vv = round(np.std(vv_series), 2)
    overall_std_vh = round(np.std(vh_series), 2)
    max_vv = max(vv_series)
    min_vv = min(vv_series)
    vv_range = round(max_vv - min_vv, 2)

    # Seasonal patterns
    spring_vv = np.mean([v for d, v in zip(dates, vv_series)
                         if d.month in [3, 4, 5]])
    summer_vv = np.mean([v for d, v in zip(dates, vv_series)
                         if d.month in [6, 7, 8]])
    autumn_vv = np.mean([v for d, v in zip(dates, vv_series)
                         if d.month in [9, 10, 11]])
    winter_vv = np.mean([v for d, v in zip(dates, vv_series)
                         if d.month in [12, 1, 2]])

    # Detect spikes
    spikes = []
    for i in range(1, len(vv_series)):
        if vv_series[i-1] > 0:
            change = vv_series[i] / vv_series[i-1]
            if change > 1.3:
                spikes.append({
                    "date": dates[i].strftime("%Y-%m-%d"),
                    "month": dates[i].month,
                    "ratio": round(change, 3)
                })

    # Detect drops
    drops = []
    for i in range(1, len(vv_series)):
        if vv_series[i-1] > 0:
            change = vv_series[i] / vv_series[i-1]
            if change < 0.75:
                drops.append({
                    "date": dates[i].strftime("%Y-%m-%d"),
                    "month": dates[i].month,
                    "ratio": round(change, 3)
                })

    return {
        "overall_mean_vv": overall_mean_vv,
        "overall_mean_vh": overall_mean_vh,
        "overall_std_vv": overall_std_vv,
        "overall_std_vh": overall_std_vh,
        "vv_range": vv_range,
        "spring_vv": round(float(spring_vv), 2) if not np.isnan(spring_vv) else None,
        "summer_vv": round(float(summer_vv), 2) if not np.isnan(summer_vv) else None,
        "autumn_vv": round(float(autumn_vv), 2) if not np.isnan(autumn_vv) else None,
        "winter_vv": round(float(winter_vv), 2) if not np.isnan(winter_vv) else None,
        "spikes": spikes,
        "drops": drops,
        "monthly_vv": avg_vv,
        "monthly_vh": avg_vh,
        "n_observations": len(dates),
        "date_range_start": dates[0].strftime("%Y-%m-%d"),
        "date_range_end": dates[-1].strftime("%Y-%m-%d")
    }


def classify_crop(observations):
    """
    Classify crop type from SAR time series
    Uses rule-based detection from seasonal patterns

    Returns crop type with confidence score
    """
    stats = get_seasonal_stats(observations)
    if not stats:
        return {"crop_type": "Unknown", "confidence": 0, "reason": "No data"}

    scores = {
        "Winter Wheat": 0,
        "Spring Barley": 0,
        "Potato": 0,
        "Oilseed Rape": 0,
        "Grassland": 0
    }

    reasons = {k: [] for k in scores}

    # --- GRASSLAND detection ---
    # Very stable signal year round, low VV range
    if stats["vv_range"] < 30:
        scores["Grassland"] += 40
        reasons["Grassland"].append("Low VV range — stable signal")
    if stats["overall_std_vv"] < 8:
        scores["Grassland"] += 30
        reasons["Grassland"].append("Low VV variability")
    if stats["overall_mean_vv"] < 40:
        scores["Grassland"] += 20
        reasons["Grassland"].append("Low mean VV — typical grassland")

    # --- WINTER WHEAT detection ---
    # High VV in autumn (sowing), spike in May-June (heading)
    june_spikes = [s for s in stats["spikes"] if s["month"] in [5, 6]]
    if june_spikes:
        scores["Winter Wheat"] += 40
        reasons["Winter Wheat"].append(
            f"Heading spike detected {june_spikes[0]['date']}")

    oct_vv = stats["monthly_vv"].get(10, 0)
    if oct_vv and oct_vv > 60:
        scores["Winter Wheat"] += 20
        reasons["Winter Wheat"].append("High October VV — autumn sowing")

    aug_drops = [d for d in stats["drops"] if d["month"] in [7, 8]]
    if aug_drops:
        scores["Winter Wheat"] += 20
        reasons["Winter Wheat"].append(
            f"Harvest drop detected {aug_drops[0]['date']}")

    # --- SPRING BARLEY detection ---
    # Low signal in winter, spike in July (heading)
    july_spikes = [s for s in stats["spikes"] if s["month"] in [7]]
    if july_spikes:
        scores["Spring Barley"] += 40
        reasons["Spring Barley"].append(
            f"Heading spike detected {july_spikes[0]['date']}")

    apr_vv = stats["monthly_vv"].get(4, 0)
    if apr_vv and apr_vv < 70:
        scores["Spring Barley"] += 15
        reasons["Spring Barley"].append("Low April VV — spring sowing")

    # --- POTATO detection ---
    # No sharp spike, gradual VH rise April-July
    # High VH/VV ratio in summer
    summer_vh = stats["monthly_vh"].get(7, 0)
    summer_vv_val = stats["monthly_vv"].get(7, 0)
    if summer_vh and summer_vv_val:
        summer_ratio = summer_vh / summer_vv_val
        if summer_ratio > 0.18:
            scores["Potato"] += 30
            reasons["Potato"].append("High summer VH/VV ratio")

    if not june_spikes and not july_spikes:
        scores["Potato"] += 20
        reasons["Potato"].append("No heading spike — consistent with potato")

    # --- OILSEED RAPE detection ---
    # Sowing August-September, flowering peak April-May
    sep_vv = stats["monthly_vv"].get(9, 0)
    if sep_vv and sep_vv > 50:
        scores["Oilseed Rape"] += 20
        reasons["Oilseed Rape"].append("High September VV — autumn sowing")

    may_vh = stats["monthly_vh"].get(5, 0)
    apr_vh = stats["monthly_vh"].get(4, 0)
    if may_vh and apr_vh:
        spring_vh_peak = max(may_vh, apr_vh)
        if spring_vh_peak > 12:
            scores["Oilseed Rape"] += 35
            reasons["Oilseed Rape"].append(
                "High spring VH — flowering canopy")

    jul_drops = [d for d in stats["drops"] if d["month"] in [7]]
    if jul_drops:
        scores["Oilseed Rape"] += 20
        reasons["Oilseed Rape"].append(
            f"July harvest drop — consistent with OSR")

    # Determine best crop
    best_crop = max(scores, key=scores.get)
    best_score = scores[best_crop]
    total = sum(scores.values())
    confidence_pct = round(best_score / total * 100) if total > 0 else 0

    return {
        "crop_type": best_crop,
        "confidence_pct": confidence_pct,
        "scores": scores,
        "reasons": reasons[best_crop],
        "all_reasons": reasons,
        "stats": stats
    }


def full_field_analysis(observations):
    """
    Complete field analysis:
    1. Classify crop type
    2. Detect planting/sowing date
    3. Detect current growth stage
    4. Generate management alerts
    5. Estimate yield potential
    """
    # Step 1 — classify crop
    classification = classify_crop(observations)
    crop_type = classification["crop_type"]

    # Step 2 — run appropriate growth model
    growth_result = None

    if crop_type == "Winter Wheat":
        growth_result = estimate_wheat_stage(observations)
    elif crop_type == "Spring Barley":
        growth_result = estimate_current_stage(observations)
    elif crop_type == "Potato":
        growth_result = estimate_potato_stage(observations)
    elif crop_type == "Oilseed Rape":
        growth_result = estimate_osr_stage(observations)

    # Step 3 — basic yield potential estimate
    # Based on development stage and seasonal progress
    yield_estimate = None
    yield_range = None

    if growth_result and not growth_result.get("error"):
        if crop_type == "Winter Wheat":
            bbch = growth_result.get("current_bbch", 0)
            if bbch >= 55:
                yield_estimate = 8.5
                yield_range = "7.5-9.5 t/ha"
            elif bbch >= 30:
                yield_estimate = 7.5
                yield_range = "6.0-9.0 t/ha"
        elif crop_type == "Spring Barley":
            bbch = growth_result.get("current_bbch", 0)
            if bbch >= 55:
                yield_estimate = 5.5
                yield_range = "4.5-6.5 t/ha"
        elif crop_type == "Potato":
            stage = growth_result.get("current_stage_id", 0)
            if stage >= 80:
                yield_estimate = 38.0
                yield_range = "30-45 t/ha"
        elif crop_type == "Oilseed Rape":
            stage = growth_result.get("current_stage_id", 0)
            if stage >= 60:
                yield_estimate = 3.8
                yield_range = "3.0-4.5 t/ha"

    return {
        "field_analysis": {
            "crop_type": crop_type,
            "classification_confidence": classification["confidence_pct"],
            "classification_reasons": classification["reasons"],
            "planting_date": (
                growth_result.get("sowing_date_detected") or
                growth_result.get("planting_date_detected")
                if growth_result else None
            ),
            "current_stage": (
                growth_result.get("current_stage") or
                growth_result.get("current_bbch")
                if growth_result else None
            ),
            "yield_estimate_tha": yield_estimate,
            "yield_range": yield_range,
            "management_alerts": (
                growth_result.get("management_alerts", [])
                if growth_result else []
            ),
            "growth_model_confidence": (
                growth_result.get("confidence")
                if growth_result else "LOW"
            )
        },
        "growth_model": growth_result,
        "classification_detail": classification
    }


if __name__ == "__main__":
    import json, os, sys
    sys.path.insert(0, '/workspaces/crop-trajectory')

    print("Testing Crop Classifier — Ireland 2026")
    print("="*50)

    test_file = "/workspaces/crop-trajectory/sar_ireland_2026.json"

    if os.path.exists(test_file):
        with open(test_file) as f:
            observations = json.load(f)

        result = full_field_analysis(observations)
        fa = result["field_analysis"]

        print(f"Crop type:          {fa['crop_type']}")
        print(f"Confidence:         {fa['classification_confidence']}%")
        print(f"Reasons:            {fa['classification_reasons']}")
        print(f"Planting date:      {fa['planting_date']}")
        print(f"Current stage:      {fa['current_stage']}")
        print(f"Yield estimate:     {fa['yield_estimate_tha']} t/ha")
        print(f"Yield range:        {fa['yield_range']}")
        print(f"Model confidence:   {fa['growth_model_confidence']}")
        if fa['management_alerts']:
            print("Management alerts:")
            for alert in fa['management_alerts']:
                print(f"  → {alert}")
