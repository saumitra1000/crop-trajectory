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


def get_optical_monthly(polygon, client_id, client_secret, token=None,
                        start_date="2025-10-01", end_date="2026-06-12"):
    """Extract monthly NDVI and NDRE via Element84 STAC + COG — free, no CDSE PU"""
    import struct, zlib
    from numpy.linalg import lstsq

    lngs = [c[0] for c in polygon]
    lats  = [c[1] for c in polygon]
    bbox  = [min(lngs), min(lats), max(lngs), max(lats)]
    lat_c = (bbox[1] + bbox[3]) / 2
    lng_c = (bbox[0] + bbox[2]) / 2

    def fetch_range(url, start, end):
        r = requests.get(url, headers={"Range": f"bytes={start}-{end-1}"}, timeout=30)
        if r.status_code not in [200, 206]:
            raise Exception(f"HTTP {r.status_code}")
        return r.content

    def read_s2_pixel(url, lat, lng):
        """Read S2 COG pixel using ModelPixelScale + tiepoint georeferencing"""
        try:
            fmt = "<"
            header = fetch_range(url, 0, 65536)
            ifd_offset = struct.unpack_from(f"{fmt}I", header, 4)[0]
            n_tags = struct.unpack_from(f"{fmt}H", header, ifd_offset)[0]
            tags = {}
            for i in range(n_tags):
                off = ifd_offset + 2 + i * 12
                if off + 12 > len(header): break
                tag = struct.unpack_from(f"{fmt}H", header, off)[0]
                typ = struct.unpack_from(f"{fmt}H", header, off+2)[0]
                cnt = struct.unpack_from(f"{fmt}I", header, off+4)[0]
                val = struct.unpack_from(f"{fmt}I", header, off+8)[0]
                tags[tag] = (typ, cnt, val)

            width  = tags[256][2]; height = tags[257][2]
            tile_w = tags[322][2]; tile_h = tags[323][2]
            tile_off_ptr  = tags[324][2]; n_tiles_count = tags[324][1]
            tile_byte_ptr = tags[325][2]

            # Try ModelPixelScale (tag 33550) first
            if 33550 in tags:
                scale_off = tags[33550][2]
                scale_data = fetch_range(url, scale_off, scale_off + 24)
                sx = struct.unpack_from(f"{fmt}d", scale_data, 0)[0]
                sy = struct.unpack_from(f"{fmt}d", scale_data, 8)[0]
                tie_off = tags[33922][2]
                tie_data = fetch_range(url, tie_off, tie_off + 48)
                tie_x = struct.unpack_from(f"{fmt}d", tie_data, 24)[0]
                tie_y = struct.unpack_from(f"{fmt}d", tie_data, 32)[0]
                col = int((lng - tie_x) / sx)
                row = int((tie_y - lat) / sy)
            else:
                # GCP-based (tag 33922 with multiple tiepoints)
                tie_off = tags[33922][2]; tie_cnt = tags[33922][1]
                tie_data = fetch_range(url, tie_off, tie_off + tie_cnt * 8)
                gcps = []
                for i in range(tie_cnt // 6):
                    o = i * 48
                    px = struct.unpack_from(f"{fmt}d", tie_data, o)[0]
                    py = struct.unpack_from(f"{fmt}d", tie_data, o+8)[0]
                    gx = struct.unpack_from(f"{fmt}d", tie_data, o+24)[0]
                    gy = struct.unpack_from(f"{fmt}d", tie_data, o+32)[0]
                    gcps.append((px, py, gx, gy))
                cols_a = np.array([g[0] for g in gcps])
                rows_a = np.array([g[1] for g in gcps])
                lngs_a = np.array([g[2] for g in gcps])
                lats_a = np.array([g[3] for g in gcps])
                A = np.column_stack([lngs_a, lats_a, np.ones(len(gcps))])
                cc, _, _, _ = lstsq(A, cols_a, rcond=None)
                rc, _, _, _ = lstsq(A, rows_a, rcond=None)
                col = int(cc[0]*lng + cc[1]*lat + cc[2])
                row = int(rc[0]*lng + rc[1]*lat + rc[2])

            if not (0 <= col < width and 0 <= row < height):
                return None

            tiles_across = (width + tile_w - 1) // tile_w
            tile_idx = (row // tile_h) * tiles_across + (col // tile_w)
            off_data  = fetch_range(url, tile_off_ptr,  tile_off_ptr  + n_tiles_count * 4)
            size_data = fetch_range(url, tile_byte_ptr, tile_byte_ptr + n_tiles_count * 4)
            t_offset = struct.unpack_from(f"{fmt}I", off_data,  tile_idx * 4)[0]
            t_size   = struct.unpack_from(f"{fmt}I", size_data, tile_idx * 4)[0]
            if t_size == 0: return None

            tile_raw = fetch_range(url, t_offset, t_offset + t_size)
            try:    raw = zlib.decompress(tile_raw)
            except:
                try: raw = zlib.decompress(tile_raw, -15)
                except: raw = tile_raw

            lc = col % tile_w; lr = row % tile_h
            bits = tags.get(258,(0,0,16))[2]
            bpp  = bits // 8
            px_off = (lr * tile_w + lc) * bpp
            if px_off + bpp > len(raw): return None
            if bpp == 2:
                dn = struct.unpack_from(f"{fmt}H", raw, px_off)[0]
            else:
                dn = struct.unpack_from(f"{fmt}B", raw, px_off)[0]
            return dn if dn > 0 else None
        except Exception:
            return None

    # Search for S2 scenes
    try:
        r = requests.post(
            "https://earth-search.aws.element84.com/v1/search",
            json={
                "collections": ["sentinel-2-l2a"],
                "bbox": bbox,
                "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
                "limit": 200
            }, timeout=30
        )
        features = r.json().get("features", [])
    except Exception:
        return {}, {}

    # Best observation per month (lowest cloud cover)
    monthly_candidates = {}

    for f in features:
        date_str = f["properties"].get("datetime","")[:10]
        if not date_str: continue
        month = int(date_str.split("-")[1])
        cloud = f["properties"].get("eo:cloud_cover", 100)
        if cloud > 85: continue  # skip very cloudy

        assets = f.get("assets", {})
        red_url = assets.get("red",{}).get("href","")
        nir_url = assets.get("nir",{}).get("href","")
        re1_url = assets.get("rededge1",{}).get("href","")
        scl_url = assets.get("scl",{}).get("href","")
        if not red_url or not nir_url: continue

        if month not in monthly_candidates or cloud < monthly_candidates[month][0]:
            monthly_candidates[month] = (cloud, red_url, nir_url, re1_url, date_str)

    monthly_ndvi = {}
    monthly_ndre = {}

    for month, (cloud, red_url, nir_url, re1_url, date_str) in monthly_candidates.items():
        red_dn = read_s2_pixel(red_url, lat_c, lng_c)
        nir_dn = read_s2_pixel(nir_url, lat_c, lng_c)
        re1_dn = read_s2_pixel(re1_url, lat_c, lng_c) if re1_url else None

        if red_dn and nir_dn and red_dn > 0:
            ndvi = (nir_dn - red_dn) / (nir_dn + red_dn + 0.0001)
            if -1 < ndvi < 1.5:
                monthly_ndvi[month] = round(ndvi, 4)
            if re1_dn and re1_dn > 0:
                ndre = (nir_dn - re1_dn) / (nir_dn + re1_dn + 0.0001)
                if -1 < ndre < 1.5:
                    monthly_ndre[month] = round(ndre, 4)

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
