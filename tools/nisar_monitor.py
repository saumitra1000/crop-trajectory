#!/usr/bin/env python3
"""
NISAR Ireland Coverage Monitor
Run weekly: python3 tools/nisar_monitor.py
Expected: data available July 2026
"""
import requests
from datetime import datetime

COLLECTIONS = [
    "NISAR_L2_GCOV_BETA_V1",
    "NISAR_L2_GCOV_V1",
    "NISAR_L1_RSLC_V1",
]
IRELAND_BBOX = "-10.5,51.3,-5.9,55.5"

print(f"NISAR Ireland Monitor — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
for col in COLLECTIONS:
    r = requests.get(
        "https://cmr.earthdata.nasa.gov/search/granules.json",
        params={"short_name": col, "bounding_box": IRELAND_BBOX,
                "page_size": 5, "sort_key": "-start_date"},
        timeout=30
    )
    granules = r.json().get("feed", {}).get("entry", [])
    if granules:
        print(f"✅ {col}: {len(granules)} granules over Ireland!")
        for g in granules[:2]:
            print(f"   {g.get('title','')[:60]}")
            print(f"   Date: {g.get('time_start','')[:10]}")
    else:
        print(f"⏳ {col}: no Ireland coverage yet")
