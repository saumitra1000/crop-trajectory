"""
Fusion Extractor — SAR + NDVI + NDRE per parcel
Used to build CropFusion training dataset

Returns:
    monthly_vh: {1..12}
    monthly_vv: {1..12}
    monthly_ndvi: {1..12}
    monthly_ndre: {1..12}
"""

import requests
import numpy as np
import time
import os


def get_token(client_id, client_secret):
    r = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={"client_id": client_id, "client_secret": client_secret,
              "grant_type": "client_credentials"}, timeout=30)
    return r.json().get("access_token")


def get_optical_monthly(polygon, client_id, client_secret, token=None, start_date="2025-10-01", end_date="2026-06-12"):
    """Extract monthly NDVI and NDRE via Sentinel-2 Statistics API"""
    if not token:
        token = get_token(client_id, client_secret)

    lngs = [c[0] for c in polygon]
    lats = [c[1] for c in polygon]
    bbox = [min(lngs), min(lats), max(lngs), max(lats)]

    # Use fixed pixel dimensions instead of resolution
    # API limit: max 1500 pixels per side
    # Target: ~128 pixels max per side for speed
    bbox_width_deg = bbox[2] - bbox[0]
    bbox_height_deg = bbox[3] - bbox[1]
    target_pixels = 64
    res_x = bbox_width_deg / target_pixels
    res_y = bbox_height_deg / target_pixels
    res = max(res_x, res_y, 10/111000)  # min 10m equiv

    # Use 5-day intervals to get all acquisitions then pick best per month
    payload = {
        "input": {
            "bounds": {"bbox": bbox,
                       "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "sentinel-2-l2a",
                      "dataFilter": {"maxCloudCoverage": 100}}]  # no cloud filter — we mask per pixel
        },
        "aggregation": {
            "timeRange": {"from": f"{start_date}T00:00:00Z",
                          "to": f"{end_date}T23:59:59Z"},
            "aggregationInterval": {"of": "P5D"},  # 5-day intervals = all acquisitions
            "evalscript": """
//VERSION=3
function setup(){
  return {
    input:[{bands:["B04","B05","B08","SCL"]}],
    output:[
      {id:"ndvi",bands:1,sampleType:"FLOAT32"},
      {id:"clear",bands:1,sampleType:"FLOAT32"},
      {id:"ndre",bands:1,sampleType:"FLOAT32"},
      {id:"dataMask",bands:1}
    ]
  };
}
function evaluatePixel(s){
  var clear=([3,8,9,10].includes(s.SCL))?0:1;
  var ndvi=(s.B08-s.B04)/(s.B08+s.B04+0.0001);
  var ndre=(s.B08-s.B05)/(s.B08+s.B05+0.0001);
  return {ndvi:[clear?ndvi:NaN], clear:[clear], ndre:[clear?ndre:NaN], dataMask:[1]};
}
            """,
            "resx": res_x, "resy": res_y
        },
        "calculations": {"default": {}}
    }

    resp = requests.post(
        "https://sh.dataspace.copernicus.eu/api/v1/statistics",
        json=payload,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
        timeout=90)

    # Aggregate to monthly — take best observation per month
    # (highest clear pixel fraction)
    monthly_candidates = {}  # month -> list of (ndvi, ndre, clear_pct)

    if resp.status_code == 200:
        for interval in resp.json().get("data", []):
            date = interval.get("interval", {}).get("from", "")[:7]
            month = int(date.split("-")[1])
            outputs = interval.get("outputs", {})
            ndvi = outputs.get("ndvi",{}).get("bands",{}).get("B0",{}).get("stats",{}).get("mean")
            clear = outputs.get("clear",{}).get("bands",{}).get("B0",{}).get("stats",{}).get("mean", 0)
            ndre = outputs.get("ndre",{}).get("bands",{}).get("B0",{}).get("stats",{}).get("mean")
            n = outputs.get("ndvi",{}).get("bands",{}).get("B0",{}).get("stats",{}).get("sampleCount", 0)

            # Only use if at least 20% of pixels are clear
            if isinstance(ndvi, float) and not np.isnan(ndvi) and isinstance(clear, float) and clear >= 0.2:
                if month not in monthly_candidates:
                    monthly_candidates[month] = []
                # ndre is B2 — validate range (should be -1 to 1, not 0/1)
                ndre_valid = ndre if (isinstance(ndre, float) and -1 < ndre < 1.01 and not np.isnan(ndre)) else None
                monthly_candidates[month].append((ndvi, ndre_valid, clear, n))

    # Pick best observation per month (highest clear fraction)
    monthly_ndvi = {}
    monthly_ndre = {}
    for month, candidates in monthly_candidates.items():
        best = max(candidates, key=lambda x: x[2])
        monthly_ndvi[month] = round(best[0], 4)
        if isinstance(best[1], float) and not np.isnan(best[1]):
            monthly_ndre[month] = round(best[1], 4)

    return monthly_ndvi, monthly_ndre


def extract_fusion_features(polygon, client_id, client_secret,
                             sar_observations=None,
                             start_date="2025-10-01", end_date="2026-06-12"):
    """
    Extract full feature vector for CropFusion classifier.
    
    Returns dict with:
        monthly_vh, monthly_vv, monthly_ndvi, monthly_ndre
        derived features
        n_sar, n_ndvi
    """
    token = get_token(client_id, client_secret)

    # SAR extraction
    if sar_observations is None:
        from extractors.sar_polygon import get_sar_timeseries_polygon
        sar_obs = get_sar_timeseries_polygon(
            polygon, start_date, end_date,
            client_id, client_secret, interval_days=12
        )
        available = [o for o in sar_obs if o.get("available")]
    else:
        available = [o for o in sar_observations if o.get("available")]

    # Monthly SAR averages
    monthly_vh_raw = {}
    monthly_vv_raw = {}
    for o in available:
        try:
            month = int(o["date"].split("-")[1])
            vh = o.get("vh") or o.get("vh_mean")
            vv = o.get("vv") or o.get("vv_mean")
            if vh:
                if month not in monthly_vh_raw:
                    monthly_vh_raw[month] = []
                monthly_vh_raw[month].append(float(vh))
            if vv:
                if month not in monthly_vv_raw:
                    monthly_vv_raw[month] = []
                monthly_vv_raw[month].append(float(vv))
        except:
            continue

    monthly_vh = {m: round(np.mean(v), 3) for m, v in monthly_vh_raw.items()}
    monthly_vv = {m: round(np.mean(v), 3) for m, v in monthly_vv_raw.items()}

    # Optical extraction
    monthly_ndvi, monthly_ndre = get_optical_monthly(polygon, client_id, client_secret, token, start_date=start_date, end_date=end_date)

    # Interpolate missing optical months
    def interpolate_monthly(monthly_dict, months=range(1,13)):
        """Linear interpolation for cloud-covered months"""
        known = {m: v for m, v in monthly_dict.items() if v}
        if not known:
            return {m: 0 for m in months}
        result = {}
        month_list = list(months)
        for m in month_list:
            if m in known:
                result[m] = known[m]
            else:
                # Find nearest known months before and after
                before = [k for k in sorted(known.keys()) if k < m]
                after  = [k for k in sorted(known.keys()) if k > m]
                if before and after:
                    m1, m2 = before[-1], after[0]
                    # Linear interpolation only between known values
                    result[m] = round(known[m1] + (known[m2]-known[m1]) * (m-m1)/(m2-m1), 4)
                else:
                    # Do not extrapolate beyond known range — leave as None
                    result[m] = None
        return result

    monthly_ndvi_interp = interpolate_monthly(monthly_ndvi)
    monthly_ndre_interp = interpolate_monthly(monthly_ndre)

    # Build feature vector
    vh_vec = [monthly_vh.get(m, 0) for m in range(1, 13)]
    vv_vec = [monthly_vv.get(m, 0) for m in range(1, 13)]
    import math
    ndvi_vec = [monthly_ndvi_interp.get(m) if monthly_ndvi_interp.get(m) is not None else float('nan') for m in range(1, 13)]
    ndre_vec = [monthly_ndre_interp.get(m) if monthly_ndre_interp.get(m) is not None else float('nan') for m in range(1, 13)]

    # Derived SAR — use ALL observations for range, not monthly averages
    # Monthly averaging reduces VV range by ~57% — kills primary discriminator
    all_vv_obs = []
    all_vh_obs = []
    for o in available:
        vv = o.get("vv") or o.get("vv_mean")
        vh = o.get("vh") or o.get("vh_mean")
        if vv: all_vv_obs.append(float(vv))
        if vh: all_vh_obs.append(float(vh))

    vv_range = round(max(all_vv_obs) - min(all_vv_obs), 2) if all_vv_obs else 0
    vh_range = round(max(all_vh_obs) - min(all_vh_obs), 2) if all_vh_obs else 0
    all_vv = [v for v in vv_vec if v > 0]
    all_vh = [v for v in vh_vec if v > 0]

    winter_vh = np.mean([monthly_vh.get(m, 0) for m in [12, 1, 2] if m in monthly_vh]) if monthly_vh else 0
    spring_vh = np.mean([monthly_vh.get(m, 0) for m in [3, 4, 5] if m in monthly_vh]) if monthly_vh else 0
    summer_vh = np.mean([monthly_vh.get(m, 0) for m in [6, 7, 8] if m in monthly_vh]) if monthly_vh else 0

    # Derived NDVI
    all_ndvi = [v for v in ndvi_vec if v > 0]
    ndvi_range = round(max(all_ndvi) - min(all_ndvi), 3) if all_ndvi else 0
    ndvi_winter = np.mean([monthly_ndvi.get(m, 0) for m in [12, 1, 2]]) 
    ndvi_spring = np.mean([monthly_ndvi.get(m, 0) for m in [3, 4, 5]])
    ndvi_summer = np.mean([monthly_ndvi.get(m, 0) for m in [6, 7, 8]])

    peak_vh_month = vh_vec.index(max(vh_vec)) + 1 if all_vh else 0
    peak_ndvi_month = ndvi_vec.index(max(ndvi_vec)) + 1 if all_ndvi else 0

    n_valid_ndvi = sum(1 for v in ndvi_vec if not (v != v))  # count non-NaN
    n_valid_ndre = sum(1 for v in ndre_vec if not (v != v))

    features = (
        vh_vec +        # 12
        vv_vec +        # 12
        ndvi_vec +      # 12
        ndre_vec +      # 12
        [               # 16 derived
            n_valid_ndvi, n_valid_ndre,
            vv_range, vh_range, ndvi_range,
            float(winter_vh), float(spring_vh), float(summer_vh),
            float(summer_vh - winter_vh),
            float(spring_vh - winter_vh),
            float(ndvi_winter), float(ndvi_spring), float(ndvi_summer),
            float(ndvi_summer - ndvi_winter),
            peak_vh_month, peak_ndvi_month
        ]
    )

    return {
        "monthly_vh": monthly_vh,
        "monthly_vv": monthly_vv,
        "monthly_ndvi": monthly_ndvi,
        "monthly_ndre": monthly_ndre,
        "features": features,
        "n_sar": len(available),
        "n_ndvi": len(monthly_ndvi),
        "n_ndre": len(monthly_ndre)
    }


if __name__ == "__main__":
    import json, sys
    sys.path.insert(0, '/workspaces/crop-trajectory')

    cid = os.environ.get('CDSE_CLIENT_ID', 'sh-6e5978f5-f5d6-43d6-874d-720d84121683')
    csec = os.environ.get('CDSE_CLIENT_SECRET', 'yrMEXQ5drlF26yrB4sTEXfWOIwKtB1fP')

    with open('/workspaces/crop-trajectory/data/irish_validation_parcels.json') as f:
        data = json.load(f)

    # Test on one parcel from each class
    test_crops = ["Permanent Pasture", "Barley - Spring", "Oilseed Rape - Winter"]

    for crop in test_crops:
        parcels = sorted(data[crop], key=lambda x: x['area_ha'], reverse=True)
        p = parcels[0]
        print(f"\nTesting {crop} — {p['area_ha']}ha")
        result = extract_fusion_features(p['polygon'], cid, csec)
        print(f"  SAR months:  {sorted(result['monthly_vh'].keys())}")
        print(f"  NDVI months: {sorted(result['monthly_ndvi'].keys())}")
        print(f"  NDRE months: {sorted(result['monthly_ndre'].keys())}")
        print(f"  Features:    {len(result['features'])}")
        print(f"  VV range:    {result['features'][48]}")
        print(f"  NDVI range:  {result['features'][50]}")
