"""
Crop Identification from SAR Time Series
Uses VH backscatter seasonal pattern to identify crop type
No reliance on DAFM declaration date

Key signatures (from 479 Irish parcels):
Grassland:    flat VH year-round, VV range < 100
Spring Barley: low winter VH, rising spring, drops at harvest
Winter Wheat:  moderate winter VH, peaks May-Jun, drops Jul
Oilseed Rape:  high winter VH, dips at flowering Mar-Apr
Oats:         highest winter VH of all crops
"""

import numpy as np
from datetime import datetime


def extract_monthly_vh(observations):
    """Extract monthly average VH from SAR time series"""
    monthly = {}
    for o in observations:
        if not o.get("available"):
            continue
        try:
            month = int(o["date"].split("-")[1])
            vh = o.get("vh") or o.get("vh_mean")
            vv = o.get("vv") or o.get("vv_mean")
            if vh:
                if month not in monthly:
                    monthly[month] = {"vh": [], "vv": []}
                monthly[month]["vh"].append(vh)
                if vv:
                    monthly[month]["vv"].append(vv)
        except:
            continue

    result = {}
    for m, vals in monthly.items():
        result[m] = {
            "vh": round(np.mean(vals["vh"]), 2),
            "vv": round(np.mean(vals["vv"]), 2) if vals["vv"] else None
        }
    return result


def analyse_time_series(observations):
    """
    Analyse SAR time series to identify crop type and growth stage.

    Returns:
        crop_type: identified crop
        confidence: 0-100
        evidence: list of evidence points
        growth_phase: current phase
        key_events: detected events (sowing, harvest, grazing)
    """
    if not observations:
        return {"crop_type": "Unknown", "confidence": 0}

    available = [o for o in observations if o.get("available")]
    if len(available) < 5:
        return {"crop_type": "Unknown", "confidence": 10,
                "evidence": ["Insufficient SAR observations"]}

    monthly = extract_monthly_vh(available)

    # Seasonal VH values
    winter_vh = np.mean([monthly[m]["vh"] for m in [12, 1, 2]
                         if m in monthly]) if any(m in monthly for m in [12, 1, 2]) else None
    spring_vh = np.mean([monthly[m]["vh"] for m in [3, 4, 5]
                         if m in monthly]) if any(m in monthly for m in [3, 4, 5]) else None
    summer_vh = np.mean([monthly[m]["vh"] for m in [6, 7, 8]
                         if m in monthly]) if any(m in monthly for m in [6, 7, 8]) else None

    # VV range — key grassland discriminator
    all_vv = [o.get("vv") or o.get("vv_mean") for o in available
              if o.get("vv") or o.get("vv_mean")]
    vv_range = round(max(all_vv) - min(all_vv), 2) if len(all_vv) > 3 else None

    # All VH values for trend analysis
    all_vh = [(o["date"], o.get("vh") or o.get("vh_mean"))
              for o in available if o.get("vh") or o.get("vh_mean")]
    all_vh.sort(key=lambda x: x[0])

    evidence = []
    events = []
    crop_scores = {
        "Grassland": 0,
        "Spring Barley": 0,
        "Winter Wheat": 0,
        "Oilseed Rape": 0,
        "Oats": 0
    }

    # === KEY DISCRIMINATORS ===

    # 1. VV range — strongest grassland signal
    if vv_range is not None:
        if vv_range < 100:
            crop_scores["Grassland"] += 40
            evidence.append(f"Low VV range ({vv_range}) — stable year-round signal (grassland)")
        elif vv_range > 180:
            for c in ["Spring Barley", "Winter Wheat", "Oilseed Rape"]:
                crop_scores[c] += 20
            evidence.append(f"High VV range ({vv_range}) — strong seasonal variation (arable)")

    # 2. Winter VH level
    if winter_vh is not None:
        if winter_vh > 38:
            crop_scores["Oats"] += 35
            evidence.append(f"Very high winter VH ({round(winter_vh,1)}) — oats signature")
        elif winter_vh > 30:
            crop_scores["Oilseed Rape"] += 20
            crop_scores["Oats"] += 10
            evidence.append(f"High winter VH ({round(winter_vh,1)}) — established crop")
        elif winter_vh < 20:
            crop_scores["Spring Barley"] += 25
            evidence.append(f"Low winter VH ({round(winter_vh,1)}) — bare soil (spring sown)")

    # 3. Spring VH trend
    if spring_vh is not None and winter_vh is not None:
        spring_rise = spring_vh - winter_vh
        if spring_rise > 10:
            crop_scores["Spring Barley"] += 20
            crop_scores["Winter Wheat"] += 15
            evidence.append(f"Rising spring VH (+{round(spring_rise,1)}) — canopy development")
            events.append({
                "event": "Canopy development",
                "period": "Spring",
                "signal": f"+{round(spring_rise,1)} VH"
            })
        elif spring_rise < -5:
            crop_scores["Oilseed Rape"] += 25
            evidence.append(f"Spring VH dip ({round(spring_rise,1)}) — flowering signature (OSR)")
            events.append({
                "event": "Flowering detected",
                "period": "March-April",
                "signal": f"{round(spring_rise,1)} VH dip"
            })

    # 4. Summer VH
    if summer_vh is not None:
        if summer_vh < 16:
            crop_scores["Spring Barley"] += 30
            crop_scores["Winter Wheat"] += 20
            evidence.append(f"Low summer VH ({round(summer_vh,1)}) — harvest or senescence")
            events.append({
                "event": "Harvest approaching",
                "period": "Summer",
                "signal": f"VH {round(summer_vh,1)}"
            })
        elif summer_vh > 40:
            crop_scores["Oats"] += 20
            crop_scores["Grassland"] += 15
            evidence.append(f"High summer VH ({round(summer_vh,1)}) — dense biomass")

    # 5. Sudden VH drops — grazing or harvest events
    vh_vals = [v for _, v in all_vh if v]
    for i in range(1, len(vh_vals)):
        drop = vh_vals[i-1] - vh_vals[i]
        if drop > 15:
            date = all_vh[i][0]
            events.append({
                "event": "Grazing or harvest event",
                "date": date,
                "signal": f"-{round(drop,1)} VH drop"
            })
            if vv_range and vv_range < 100:
                evidence.append(f"Sudden VH drop on {date} — likely grazing event")
            else:
                evidence.append(f"Sudden VH drop on {date} — possible harvest")

    # === DETERMINE CROP TYPE ===
    best_crop = max(crop_scores, key=crop_scores.get)
    best_score = crop_scores[best_crop]
    total = sum(crop_scores.values())

    if total > 0:
        confidence = min(95, round(best_score / total * 150))
    else:
        confidence = 20

    # Determine current growth phase
    current_month = datetime.now().month
    growth_phase = determine_growth_phase(best_crop, current_month,
                                          monthly, spring_vh, summer_vh)

    return {
        "crop_type": best_crop,
        "confidence": confidence,
        "evidence": evidence,
        "growth_phase": growth_phase,
        "key_events": events,
        "scores": crop_scores,
        "seasonal_vh": {
            "winter": round(float(winter_vh), 2) if winter_vh else None,
            "spring": round(float(spring_vh), 2) if spring_vh else None,
            "summer": round(float(summer_vh), 2) if summer_vh else None
        },
        "vv_range": vv_range,
        "method": "SAR time series analysis"
    }


def determine_growth_phase(crop, month, monthly, spring_vh, summer_vh):
    """Determine current growth phase from crop and month"""
    if crop == "Grassland":
        if month in [3, 4, 5, 6]:
            return "Active growth — peak season"
        elif month in [7, 8, 9]:
            return "Summer growth — monitor cover"
        elif month in [10, 11]:
            return "Autumn flush — reducing"
        else:
            return "Winter — slow growth"

    elif crop == "Spring Barley":
        if month in [3, 4]:
            return "Sowing / Germination"
        elif month in [5, 6]:
            return "Stem Extension / Tillering"
        elif month == 7:
            return "Heading / Ear Emergence"
        elif month == 8:
            return "Ripening / Harvest"
        else:
            return "Bare soil / Stubble"

    elif crop == "Winter Wheat":
        if month in [10, 11]:
            return "Establishment / Tillering"
        elif month in [12, 1, 2]:
            return "Vernalisation / Dormancy"
        elif month in [3, 4]:
            return "Stem Extension"
        elif month in [5, 6]:
            return "Heading / Ear Emergence"
        elif month == 7:
            return "Ripening / Harvest"
        else:
            return "Bare soil / Post-harvest"

    elif crop == "Oilseed Rape":
        if month in [9, 10]:
            return "Establishment"
        elif month in [11, 12, 1, 2]:
            return "Rosette / Overwintering"
        elif month in [3, 4]:
            return "Flowering"
        elif month in [5, 6]:
            return "Pod Fill"
        elif month == 7:
            return "Harvest"
        else:
            return "Bare soil"

    elif crop == "Oats":
        if month in [3, 4]:
            return "Sowing / Germination"
        elif month in [5, 6]:
            return "Tillering / Stem Extension"
        elif month == 7:
            return "Heading"
        elif month == 8:
            return "Harvest"
        else:
            return "Bare soil / Stubble"

    return "Growing"


if __name__ == "__main__":
    import json
    import os
    import sys
    sys.path.insert(0, '/workspaces/crop-trajectory')

    os.environ['CDSE_CLIENT_ID'] = 'sh-6e5978f5-f5d6-43d6-874d-720d84121683'
    os.environ['CDSE_CLIENT_SECRET'] = 'yrMEXQ5drlF26yrB4sTEXfWOIwKtB1fP'

    print("Testing SAR Time Series Crop Identifier")
    print("=" * 50)

    with open('/workspaces/crop-trajectory/sar_ireland_2026.json') as f:
        obs = json.load(f)

    result = analyse_time_series(obs)
    print(f"Crop type:    {result['crop_type']}")
    print(f"Confidence:   {result['confidence']}%")
    print(f"Growth phase: {result['growth_phase']}")
    print(f"\nEvidence:")
    for e in result['evidence']:
        print(f"  → {e}")
    print(f"\nKey events:")
    for e in result['key_events']:
        print(f"  → {e}")
    print(f"\nSeasonal VH: {result['seasonal_vh']}")
    print(f"VV range:    {result['vv_range']}")
