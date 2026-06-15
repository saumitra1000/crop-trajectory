"""
ECOSTRESS Level-3 Evapotranspiration via NASA AppEEARS API
Queries ET data for parcel polygons from crop-trajectory dataset

ECOSTRESS L3 products:
  ECO3ETPTJPL — PT-JPL ET model (recommended)
  ECO3ETALEXIU — ALEXI-DisALEXI ET model

AppEEARS docs: https://appeears.earthdatacloud.nasa.gov/api/

Usage:
    python3 tools/ecostress_et.py --username NASA_USER --password NASA_PASS
    python3 tools/ecostress_et.py --token YOUR_TOKEN --parcel_id IE123456
"""

import os
import sys
import json
import time
import argparse
import requests
from datetime import datetime, timedelta

# AppEEARS base URL
API_URL = "https://appeears.earthdatacloud.nasa.gov/api"

# ECOSTRESS ET product layers
ET_PRODUCTS = [
    {
        "ProductAndVersion": "ECO3ETPTJPL.001",
        "Layer": "ETinst"          # Instantaneous ET (W/m²)
    },
    {
        "ProductAndVersion": "ECO3ETPTJPL.001",
        "Layer": "ETinstUncertainty"  # ET uncertainty
    },
    {
        "ProductAndVersion": "ECO3ETPTJPL.001",
        "Layer": "ETdaily"         # Daily ET (mm/day)
    },
]


def get_token(username, password):
    """Authenticate with NASA Earthdata and get AppEEARS token"""
    r = requests.post(
        f"{API_URL}/login",
        auth=(username, password),
        timeout=30
    )
    if r.status_code != 200:
        raise Exception(f"Login failed: {r.status_code} {r.text}")
    token = r.json().get("token")
    print(f"✅ Authenticated — token: {token[:20]}...")
    return token


def submit_area_request(token, parcel_id, polygon, start_date, end_date,
                         products=ET_PRODUCTS):
    """
    Submit an area sample request for a parcel polygon.
    AppEEARS accepts GeoJSON polygon for spatial subsetting.
    """
    # Build GeoJSON feature
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon]
            },
            "properties": {
                "id": parcel_id
            }
        }]
    }

    task = {
        "task_type": "area",
        "task_name": f"ecostress_et_{parcel_id}_{start_date}",
        "params": {
            "dates": [{
                "startDate": start_date,   # MM-DD-YYYY
                "endDate": end_date
            }],
            "layers": products,
            "output": {
                "format": {"type": "geotiff"},
                "projection": "geographic"
            },
            "geo": geojson
        }
    }

    r = requests.post(
        f"{API_URL}/task",
        json=task,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60
    )
    if r.status_code not in [200, 202]:
        raise Exception(f"Task submission failed: {r.status_code} {r.text[:200]}")

    task_id = r.json().get("task_id")
    print(f"  Submitted task {task_id} for parcel {parcel_id}")
    return task_id


def submit_point_request(token, parcel_id, lat, lng, start_date, end_date,
                          products=ET_PRODUCTS):
    """
    Submit a point sample request (faster than area for single parcels).
    Returns time series CSV.
    """
    task = {
        "task_type": "point",
        "task_name": f"ecostress_pt_{parcel_id}_{start_date}",
        "params": {
            "dates": [{
                "startDate": start_date,
                "endDate": end_date
            }],
            "layers": products,
            "coordinates": [{
                "id": parcel_id,
                "latitude": lat,
                "longitude": lng,
                "category": parcel_id
            }]
        }
    }

    r = requests.post(
        f"{API_URL}/task",
        json=task,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60
    )
    if r.status_code not in [200, 202]:
        raise Exception(f"Task failed: {r.status_code} {r.text[:200]}")

    task_id = r.json().get("task_id")
    print(f"  Submitted point task {task_id}")
    return task_id


def wait_for_task(token, task_id, poll_interval=30, timeout=1800):
    """Poll task status until done or timeout"""
    start = time.time()
    print(f"  Waiting for task {task_id}...", end="", flush=True)

    while time.time() - start < timeout:
        r = requests.get(
            f"{API_URL}/task/{task_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )
        status = r.json().get("status", "unknown")

        if status == "done":
            print(" ✅ done")
            return True
        elif status == "error":
            print(f" ❌ error: {r.json()}")
            return False
        else:
            print(".", end="", flush=True)
            time.sleep(poll_interval)

    print(" ⏰ timeout")
    return False


def download_results(token, task_id, output_dir):
    """Download task output files"""
    r = requests.get(
        f"{API_URL}/bundle/{task_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    files = r.json().get("files", [])

    os.makedirs(output_dir, exist_ok=True)
    downloaded = []

    for f in files:
        file_id = f["file_id"]
        filename = f["file_name"]
        if filename.endswith(".csv") or filename.endswith(".tif"):
            url = f"{API_URL}/bundle/{task_id}/{file_id}"
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                stream=True,
                timeout=120
            )
            filepath = os.path.join(output_dir, os.path.basename(filename))
            with open(filepath, "wb") as out:
                for chunk in resp.iter_content(chunk_size=8192):
                    out.write(chunk)
            downloaded.append(filepath)
            print(f"  Downloaded: {os.path.basename(filepath)}")

    return downloaded


def parse_et_csv(csv_path, parcel_id):
    """Parse AppEEARS point CSV to extract ET time series"""
    import csv
    results = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ID") != parcel_id:
                continue
            try:
                date_str = row.get("Date", "")
                et_inst = float(row.get("ETinst", "nan"))
                et_daily = float(row.get("ETdaily", "nan"))
                results.append({
                    "date": date_str,
                    "et_instantaneous_wm2": et_inst,
                    "et_daily_mm": et_daily,
                    "source": "ECOSTRESS ECO3ETPTJPL.001"
                })
            except (ValueError, KeyError):
                continue

    return results


def get_et_for_parcel(token, parcel_id, lat, lng, polygon=None,
                       start_date="01-01-2025", end_date="06-30-2026",
                       output_dir="data/ecostress", use_area=False):
    """
    Full pipeline: submit → wait → download → parse ET for one parcel.

    Args:
        use_area: True for GeoTIFF spatial subset, False for point time series CSV
    """
    print(f"\nProcessing parcel {parcel_id} ({lat:.4f}, {lng:.4f})")

    if use_area and polygon:
        task_id = submit_area_request(
            token, parcel_id, polygon, start_date, end_date
        )
    else:
        task_id = submit_point_request(
            token, parcel_id, lat, lng, start_date, end_date
        )

    success = wait_for_task(token, task_id)
    if not success:
        return None

    files = download_results(token, task_id, output_dir)

    # Parse CSV for point requests
    et_data = []
    for f in files:
        if f.endswith(".csv"):
            et_data = parse_et_csv(f, parcel_id)

    print(f"  ET observations: {len(et_data)}")
    return {
        "parcel_id": parcel_id,
        "lat": lat,
        "lng": lng,
        "task_id": task_id,
        "et_timeseries": et_data,
        "files": files
    }


def check_ecostress_availability(token, lat, lng):
    """Check available ECOSTRESS products and date ranges for a location"""
    r = requests.get(
        f"{API_URL}/product",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    products = r.json()
    eco = [p for p in products if "ECO" in p.get("ProductAndVersion","")]
    print(f"\nAvailable ECOSTRESS products ({len(eco)}):")
    for p in eco:
        print(f"  {p['ProductAndVersion']:<25} {p.get('Description','')[:60]}")
    return eco


def main():
    parser = argparse.ArgumentParser(description="Query ECOSTRESS ET via AppEEARS")
    parser.add_argument("--username", help="NASA Earthdata username")
    parser.add_argument("--password", help="NASA Earthdata password")
    parser.add_argument("--token",    help="Existing AppEEARS token")
    parser.add_argument("--parcel_id", default=None, help="Specific parcel ID to query")
    parser.add_argument("--start",    default="01-01-2025", help="Start date MM-DD-YYYY")
    parser.add_argument("--end",      default="06-30-2026", help="End date MM-DD-YYYY")
    parser.add_argument("--n_parcels", type=int, default=5,
                        help="Number of parcels to process from validation set")
    parser.add_argument("--output",   default="data/ecostress", help="Output directory")
    parser.add_argument("--list_products", action="store_true",
                        help="List available ECOSTRESS products and exit")
    args = parser.parse_args()

    # Authenticate
    if args.token:
        token = args.token
        print(f"Using provided token: {token[:20]}...")
    elif args.username and args.password:
        token = get_token(args.username, args.password)
    else:
        print("ERROR: Provide --token or --username + --password")
        print("\nGet NASA Earthdata account: https://urs.earthdata.nasa.gov/")
        print("AppEEARS docs: https://appeears.earthdatacloud.nasa.gov/api/")
        sys.exit(1)

    # List products
    if args.list_products:
        check_ecostress_availability(token, 53.0, -7.0)
        return

    # Load parcels
    parcel_file = "/workspaces/crop-trajectory/data/irish_validation_parcels.json"
    with open(parcel_file) as f:
        validation_data = json.load(f)

    # Select parcels to process
    parcels_to_process = []
    for crop, parcels in validation_data.items():
        for p in parcels[:max(1, args.n_parcels // 5)]:
            parcels_to_process.append({
                "parcel_id": p["par_lab"],
                "crop": crop,
                "lat": p["centroid_lat"],
                "lng": p["centroid_lng"],
                "polygon": p["polygon"],
                "area_ha": p["area_ha"]
            })
            if args.parcel_id and p["par_lab"] != args.parcel_id:
                continue

    if args.parcel_id:
        parcels_to_process = [p for p in parcels_to_process
                              if p["parcel_id"] == args.parcel_id]

    print(f"\nQuerying ECOSTRESS ET for {len(parcels_to_process)} parcels")
    print(f"Date range: {args.start} → {args.end}")
    print(f"Output: {args.output}")

    results = []
    for p in parcels_to_process:
        try:
            result = get_et_for_parcel(
                token=token,
                parcel_id=p["parcel_id"],
                lat=p["lat"],
                lng=p["lng"],
                polygon=p["polygon"],
                start_date=args.start,
                end_date=args.end,
                output_dir=args.output,
                use_area=False  # point request for speed
            )
            if result:
                result["crop"] = p["crop"]
                result["area_ha"] = p["area_ha"]
                results.append(result)

                # Print ET summary
                et_vals = [o["et_daily_mm"] for o in result["et_timeseries"]
                           if o["et_daily_mm"] == o["et_daily_mm"]]
                if et_vals:
                    import numpy as np
                    print(f"  ET daily: mean={np.mean(et_vals):.2f} "
                          f"min={min(et_vals):.2f} max={max(et_vals):.2f} mm/day")

        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(2)

    # Save results
    os.makedirs(args.output, exist_ok=True)
    out_file = os.path.join(args.output, "ecostress_et_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved {len(results)} parcel results to {out_file}")

    # Summary table
    if results:
        print(f"\n{'Parcel':<15} {'Crop':<25} {'ET obs':>8} {'Mean ET':>10}")
        print("-"*62)
        for r in results:
            et_vals = [o["et_daily_mm"] for o in r["et_timeseries"]
                      if o["et_daily_mm"] == o["et_daily_mm"]]
            mean_et = round(sum(et_vals)/len(et_vals), 2) if et_vals else "—"
            print(f"{r['parcel_id'][:14]:<15} {r['crop'][:24]:<25} "
                  f"{len(et_vals):>8} {str(mean_et):>10}")


if __name__ == "__main__":
    main()
