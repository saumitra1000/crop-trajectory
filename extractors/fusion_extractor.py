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
              "grant_type": "client_credentials"}, timeout=15)
    return r.json().get("access_token")


def get_optical_monthly(polygon, client_id, client_secret, token=None,
                        start_date="2025-10-01", end_date="2026-06-12"):
    """Extract monthly NDVI and NDRE via Element84 STAC + S2 COG — free"""
    import struct, zlib, math

    def latlon_to_utm(lat, lng):
        """WGS84 to UTM — auto zone for Ireland (29N)"""
        a = 6378137.0; f = 1/298.257223563
        b = a*(1-f); e2 = 1-(b/a)**2
        k0 = 0.9996
        # Zone from longitude
        zone = int((lng + 180) / 6) + 1
        lon0 = math.radians((zone - 1) * 6 - 180 + 3)
        lat_r = math.radians(lat); lng_r = math.radians(lng)
        N = a/math.sqrt(1-e2*math.sin(lat_r)**2)
        T = math.tan(lat_r)**2
        A_ = math.cos(lat_r)*(lng_r-lon0)
        M = a*((1-e2/4-3*e2**2/64)*lat_r
               -(3*e2/8+3*e2**2/32)*math.sin(2*lat_r)
               +(15*e2**2/256)*math.sin(4*lat_r))
        x = k0*N*(A_+(1-T)*A_**3/6) + 500000
        y = k0*(M+N*math.tan(lat_r)*(A_**2/2))
        return x, y

    def fetch_range(url, s, e):
        r = requests.get(url, headers={"Range": f"bytes={s}-{e-1}"}, timeout=15)
        if r.status_code not in [200, 206]:
            raise Exception(f"HTTP {r.status_code}")
        return r.content

    def read_s2_pixel(url, lat, lng):
        try:
            fmt = "<"
            header = fetch_range(url, 0, 65536)
            ifd_offset = struct.unpack_from(f"{fmt}I", header, 4)[0]
            n_tags = struct.unpack_from(f"{fmt}H", header, ifd_offset)[0]
            tags = {}
            for i in range(n_tags):
                off = ifd_offset + 2 + i*12
                if off+12 > len(header): break
                tag = struct.unpack_from(f"{fmt}H", header, off)[0]
                typ = struct.unpack_from(f"{fmt}H", header, off+2)[0]
                cnt = struct.unpack_from(f"{fmt}I", header, off+4)[0]
                val = struct.unpack_from(f"{fmt}I", header, off+8)[0]
                tags[tag] = (typ, cnt, val)

            width  = tags[256][2]; height = tags[257][2]
            tile_w = tags[322][2]; tile_h = tags[323][2]
            tile_off_ptr  = tags[324][2]; n_tiles = tags[324][1]
            tile_byte_ptr = tags[325][2]
            tag_type = tags[324][0]
            bpe = 8 if tag_type == 16 else 4
            fmt_c = "Q" if tag_type == 16 else "I"

            # UTM georef
            ux, uy = latlon_to_utm(lat, lng)
            scale_data = fetch_range(url, tags[33550][2], tags[33550][2]+24)
            sx = struct.unpack_from(f"{fmt}d", scale_data, 0)[0]
            sy = struct.unpack_from(f"{fmt}d", scale_data, 8)[0]
            tie_data = fetch_range(url, tags[33922][2], tags[33922][2]+48)
            tie_x = struct.unpack_from(f"{fmt}d", tie_data, 24)[0]
            tie_y = struct.unpack_from(f"{fmt}d", tie_data, 32)[0]

            col = int((ux - tie_x) / sx)
            row = int((tie_y - uy) / sy)
            if not (0 <= col < width and 0 <= row < height):
                return None

            tiles_across = (width+tile_w-1)//tile_w
            tile_idx = (row//tile_h)*tiles_across + (col//tile_w)

            off_d  = fetch_range(url, tile_off_ptr,  tile_off_ptr  + n_tiles*bpe)
            size_d = fetch_range(url, tile_byte_ptr, tile_byte_ptr + n_tiles*bpe)
            t_off  = struct.unpack_from(f"{fmt}{fmt_c}", off_d,  tile_idx*bpe)[0]
            t_size = struct.unpack_from(f"{fmt}{fmt_c}", size_d, tile_idx*bpe)[0]
            if t_size == 0 or t_size > 10_000_000: return None

            tile_raw = fetch_range(url, t_off, t_off+t_size)
            try:    raw = zlib.decompress(tile_raw)
            except:
                try: raw = zlib.decompress(tile_raw, -15)
                except: raw = tile_raw

            lc = col%tile_w; lr = row%tile_h
            px_off = (lr*tile_w+lc)*2
            if px_off+2 > len(raw): return None
            dn = struct.unpack_from(f"{fmt}H", raw, px_off)[0]
            # Sample 3x3 and return median of valid DNs
            valid_dns = []
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    lc2=(col+dc)%tile_w; lr2=(row+dr)%tile_h
                    px2=(lr2*tile_w+lc2)*2
                    if px2+2<=len(raw):
                        dn2=struct.unpack_from(f"{fmt}H",raw,px2)[0]
                        if 5 < dn2 < 10000:
                            valid_dns.append(dn2)
            if not valid_dns: return None
            import statistics
            return int(statistics.median(valid_dns))
        except Exception:
            return None

    lngs = [c[0] for c in polygon]
    lats  = [c[1] for c in polygon]
    bbox  = [min(lngs), min(lats), max(lngs), max(lats)]
    lat_c = (bbox[1]+bbox[3])/2
    lng_c = (bbox[0]+bbox[2])/2
    # Sample polygon vertices (evenly spaced) + centroid
    step = max(1, len(polygon)//5)  # max 5 sample points
    sample_pts = [(polygon[i][1], polygon[i][0]) for i in range(0, len(polygon), step)]
    sample_pts.append((lat_c, lng_c))
    sample_pts = sample_pts[:6]  # max 6 points

    try:
        r = requests.post(
            "https://earth-search.aws.element84.com/v1/search",
            json={"collections":["sentinel-2-l2a"],
                  "bbox": bbox,
                  "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
                  "limit": 200},
            timeout=15
        )
        features = r.json().get("features", [])
    except Exception:
        return {}, {}

    # Best (lowest cloud) per month — filter by margin to avoid swath edges
    monthly_candidates = {}
    for f in features:
        date_str = f["properties"].get("datetime","")[:10]
        if not date_str: continue
        month = int(date_str.split("-")[1])
        cloud = f["properties"].get("eo:cloud_cover", 100)
        if cloud > 50: continue
        # Check parcel centroid is well inside scene bbox
        scene_bbox = f.get("bbox", [])
        if scene_bbox:
            margin = min(lat_c-scene_bbox[1], scene_bbox[3]-lat_c,
                        lng_c-scene_bbox[0], scene_bbox[2]-lng_c)
            if margin < 0.05: continue
        assets = f.get("assets",{})
        red_url = assets.get("red",{}).get("href","")
        nir_url = assets.get("nir",{}).get("href","")
        re1_url = assets.get("rededge1",{}).get("href","")
        if not red_url or not nir_url: continue
        if month not in monthly_candidates or cloud < monthly_candidates[month][0]:
            monthly_candidates[month] = (cloud, red_url, nir_url, re1_url)

    monthly_ndvi = {}
    monthly_ndre = {}
    import statistics as _stats
    for month, (cloud, red_url, nir_url, re1_url) in monthly_candidates.items():
        ndvi_vals = []
        ndre_vals = []
        for slat, slng in sample_pts:
            red = read_s2_pixel(red_url, slat, slng)
            nir = read_s2_pixel(nir_url, slat, slng)
            re1 = read_s2_pixel(re1_url, slat, slng) if re1_url else None
            if red and nir and 0 < red < 60000 and 0 < nir < 60000:
                ndvi = (nir-red)/(nir+red+0.0001)
                if -0.5 < ndvi < 1.0:
                    ndvi_vals.append(ndvi)
                if re1 and 0 < re1 < 60000:
                    ndre = (nir-re1)/(nir+re1+0.0001)
                    if -0.5 < ndre < 1.0:
                        ndre_vals.append(ndre)
        if ndvi_vals:
            med = _stats.median(ndvi_vals)
            if -0.5 < med < 1.0:
                monthly_ndvi[month] = round(med, 4)
        if ndre_vals:
            med = _stats.median(ndre_vals)
            if -0.5 < med < 1.0:
                monthly_ndre[month] = round(med, 4)

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
