"""
Build CropFusion training dataset
Extracts SAR + NDVI + NDRE for 250 DAFM-labelled parcels
Runs overnight — takes 3-4 hours

Output: data/crop_fusion_dataset.json
"""

import json
import os
import sys
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, '/workspaces/crop-trajectory')

os.environ['CDSE_CLIENT_ID'] = 'sh-6e5978f5-f5d6-43d6-874d-720d84121683'
os.environ['CDSE_CLIENT_SECRET'] = 'yrMEXQ5drlF26yrB4sTEXfWOIwKtB1fP'

from extractors.sar_polygon import get_sar_timeseries_polygon

LABEL_MAP = {
    "Permanent Pasture": "Grassland",
    "Barley - Spring": "Spring Barley",
    "Oilseed Rape - Winter": "Oilseed Rape",
    "Wheat - Winter": "Winter Wheat",
    "Oats - Spring": "Oats",
}

def get_ndvi_monthly(polygon, client_id, client_secret):
    """Get monthly NDVI and NDRE for a polygon via Sentinel-2"""
    try:
        import requests
        from datetime import datetime, timedelta

        # Get Sentinel-2 token
        token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        r = requests.post(token_url, data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials"
        }, timeout=30)
        token = r.json().get("access_token")
        if not token:
            return {}

        # Calculate bbox
        lngs = [c[0] for c in polygon]
        lats = [c[1] for c in polygon]
        bbox = [min(lngs), min(lats), max(lngs), max(lats)]

        # Monthly NDVI Oct 2025 → Jun 2026
        monthly_ndvi = {}
        monthly_ndre = {}

        months = [
            ("2025-10-01", "2025-10-31", 10),
            ("2025-11-01", "2025-11-30", 11),
            ("2025-12-01", "2025-12-31", 12),
            ("2026-01-01", "2026-01-31", 1),
            ("2026-02-01", "2026-02-28", 2),
            ("2026-03-01", "2026-03-31", 3),
            ("2026-04-01", "2026-04-30", 4),
            ("2026-05-01", "2026-05-31", 5),
            ("2026-06-01", "2026-06-13", 6),
        ]

        for start, end, month in months:
            url = "https://sh.dataspace.copernicus.eu/api/v1/process"
            payload = {
                "input": {
                    "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                    "data": [{"type": "sentinel-2-l2a", "dataFilter": {
                        "timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"},
                        "maxCloudCoverage": 50
                    }}]
                },
                "output": {"width": 32, "height": 32, "responses": [{"identifier": "default", "format": {"type": "application/json"}}]},
                "evalscript": """
                    //VERSION=3
                    function setup(){return{input:[{bands:["B04","B05","B08","SCL"]}],output:{bands:2}};}
                    function evaluatePixel(s){
                        if(s.SCL==3||s.SCL==8||s.SCL==9||s.SCL==10) return [-1,-1];
                        var ndvi=(s.B08-s.B04)/(s.B08+s.B04+0.0001);
                        var ndre=(s.B08-s.B05)/(s.B08+s.B05+0.0001);
                        return [ndvi,ndre];
                    }
                    function updateOutputMetadata(sources,collections,outputMetadata){
                        var vals=[];var ndrevals=[];
                        for(var i=0;i<sources.default.tiles.length;i++){
                            var t=sources.default.tiles[i];
                            for(var j=0;j<t.data.length;j++){
                                if(t.data[j][0]>-0.5){vals.push(t.data[j][0]);ndrevals.push(t.data[j][1]);}
                            }
                        }
                        outputMetadata.userData={
                            ndvi:vals.length?vals.reduce(function(a,b){return a+b;})/vals.length:null,
                            ndre:ndrevals.length?ndrevals.reduce(function(a,b){return a+b;})/ndrevals.length:null,
                            n:vals.length
                        };
                    }
                """
            }
            try:
                resp = requests.post(url, json=payload,
                    headers={"Authorization": f"Bearer {token}",
                             "Content-Type": "application/json"},
                    timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    ud = data.get("userData", {})
                    if ud.get("ndvi") is not None:
                        monthly_ndvi[month] = round(ud["ndvi"], 4)
                    if ud.get("ndre") is not None:
                        monthly_ndre[month] = round(ud["ndre"], 4)
            except:
                pass
            time.sleep(0.2)

        return {"ndvi": monthly_ndvi, "ndre": monthly_ndre}
    except Exception as e:
        return {}


def extract_features(sar_obs, ndvi_data):
    """Extract 55-feature vector for CropFusion"""
    # Monthly SAR
    monthly_vh = {}
    monthly_vv = {}
    for o in sar_obs:
        if not o.get("available"): continue
        try:
            month = int(o["date"].split("-")[1])
            vh = o.get("vh") or o.get("vh_mean")
            vv = o.get("vv") or o.get("vv_mean")
            if vh:
                if month not in monthly_vh: monthly_vh[month] = []
                monthly_vh[month].append(float(vh))
            if vv:
                if month not in monthly_vv: monthly_vv[month] = []
                monthly_vv[month].append(float(vv))
        except: continue

    avg_vh = {m: np.mean(v) for m, v in monthly_vh.items()}
    avg_vv = {m: np.mean(v) for m, v in monthly_vv.items()}

    vh_monthly = [avg_vh.get(m, 0) for m in range(1, 13)]
    vv_monthly = [avg_vv.get(m, 0) for m in range(1, 13)]
    ndvi_monthly = [ndvi_data.get("ndvi", {}).get(m, 0) for m in range(1, 13)]
    ndre_monthly = [ndvi_data.get("ndre", {}).get(m, 0) for m in range(1, 13)]

    # Derived
    all_vv = [v for v in vv_monthly if v > 0]
    all_vh = [v for v in vh_monthly if v > 0]
    all_ndvi = [v for v in ndvi_monthly if v > 0]

    vv_range = max(all_vv) - min(all_vv) if all_vv else 0
    vh_range = max(all_vh) - min(all_vh) if all_vh else 0
    ndvi_range = max(all_ndvi) - min(all_ndvi) if all_ndvi else 0

    winter_vh = np.mean([avg_vh.get(m, 0) for m in [12, 1, 2] if m in avg_vh]) if avg_vh else 0
    spring_vh = np.mean([avg_vh.get(m, 0) for m in [3, 4, 5] if m in avg_vh]) if avg_vh else 0
    summer_vh = np.mean([avg_vh.get(m, 0) for m in [6, 7, 8] if m in avg_vh]) if avg_vh else 0

    ndvi_winter = np.mean([ndvi_data.get("ndvi", {}).get(m, 0) for m in [12, 1, 2]])
    ndvi_spring = np.mean([ndvi_data.get("ndvi", {}).get(m, 0) for m in [3, 4, 5]])
    ndvi_summer = np.mean([ndvi_data.get("ndvi", {}).get(m, 0) for m in [6, 7, 8]])

    peak_ndvi_month = ndvi_monthly.index(max(ndvi_monthly)) + 1 if all_ndvi else 0
    peak_vh_month = vh_monthly.index(max(vh_monthly)) + 1 if all_vh else 0

    features = (
        vh_monthly +       # 12
        vv_monthly +       # 12
        ndvi_monthly +     # 12
        ndre_monthly +     # 12
        [
            vv_range,
            vh_range,
            ndvi_range,
            winter_vh,
            spring_vh,
            summer_vh,
            summer_vh - winter_vh,
            spring_vh - winter_vh,
            ndvi_winter,
            ndvi_spring,
            ndvi_summer,
            ndvi_summer - ndvi_winter,
            peak_ndvi_month,
            peak_vh_month,
        ]
    )
    return features


if __name__ == "__main__":
    with open('/workspaces/crop-trajectory/data/irish_validation_parcels.json') as f:
        validation_data = json.load(f)

    # Load existing progress if any
    output_file = '/workspaces/crop-trajectory/data/crop_fusion_dataset.json'
    if os.path.exists(output_file):
        with open(output_file) as f:
            dataset = json.load(f)
        processed = set(d['par_lab'] for d in dataset)
        print(f"Resuming — {len(dataset)} already processed")
    else:
        dataset = []
        processed = set()

    total_parcels = sum(len(v) for v in validation_data.values())
    done = len(processed)

    print(f"CropFusion Dataset Builder")
    print(f"Total parcels: {total_parcels}")
    print(f"Already done: {done}")
    print(f"Remaining: {total_parcels - done}")
    print("="*50)

    for dafm_crop, parcels in validation_data.items():
        label = LABEL_MAP.get(dafm_crop)
        if not label: continue

        class_done = sum(1 for d in dataset if d['label'] == label)
        print(f"\n{dafm_crop} → {label} ({class_done}/{len(parcels)} done)")

        for i, parcel in enumerate(parcels):
            if parcel['par_lab'] in processed:
                continue

            print(f"  [{i+1}/{len(parcels)}] {parcel['area_ha']}ha...", end=" ", flush=True)

            try:
                # SAR extraction
                sar_obs = get_sar_timeseries_polygon(
                    parcel["polygon"],
                    "2025-10-01", "2026-06-12",
                    os.environ["CDSE_CLIENT_ID"],
                    os.environ["CDSE_CLIENT_SECRET"],
                    interval_days=12
                )
                available = [o for o in sar_obs if o.get("available")]

                if len(available) < 6:
                    print("insufficient SAR")
                    continue

                # NDVI/NDRE extraction
                ndvi_data = get_ndvi_monthly(
                    parcel["polygon"],
                    os.environ["CDSE_CLIENT_ID"],
                    os.environ["CDSE_CLIENT_SECRET"]
                )

                # Extract features
                features = extract_features(available, ndvi_data)

                dataset.append({
                    "par_lab": parcel["par_lab"],
                    "label": label,
                    "dafm_crop": dafm_crop,
                    "area_ha": parcel["area_ha"],
                    "features": features,
                    "n_sar": len(available),
                    "n_ndvi": len(ndvi_data.get("ndvi", {}))
                })
                processed.add(parcel["par_lab"])
                print(f"✅ SAR:{len(available)} NDVI:{len(ndvi_data.get('ndvi',{}))}")

                # Save progress every 10 parcels
                if len(dataset) % 10 == 0:
                    with open(output_file, 'w') as f:
                        json.dump(dataset, f)
                    print(f"  → Saved {len(dataset)} parcels")

                time.sleep(1)

            except Exception as e:
                print(f"Error: {e}")

    # Final save
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print("\n" + "="*50)
    print("COMPLETE")
    from collections import Counter
    labels = Counter(d['label'] for d in dataset)
    for label, count in sorted(labels.items()):
        print(f"  {label}: {count} parcels")
    print(f"  Total: {len(dataset)} parcels")
    print(f"  Features per parcel: {len(dataset[0]['features']) if dataset else 0}")
    print(f"\nSaved to {output_file}")
