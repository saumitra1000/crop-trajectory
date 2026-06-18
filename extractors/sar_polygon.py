"""
SAR Polygon Extractor — Element84 STAC + COG (GCP georeferencing)
Free — no CDSE processing units required
Direct HTTP range reads from S3 COGs
"""

import requests
import numpy as np
import struct
import zlib
from numpy.linalg import lstsq


STAC_URL = "https://earth-search.aws.element84.com/v1"


def polygon_to_bbox(polygon_coords):
    lngs = [p[0] for p in polygon_coords]
    lats  = [p[1] for p in polygon_coords]
    return [min(lngs), min(lats), max(lngs), max(lats)]


def s3_to_https(url):
    if url.startswith("s3://"):
        parts = url[5:].split("/", 1)
        return f"https://{parts[0]}.s3.amazonaws.com/{parts[1]}"
    return url


def fetch_range(url, start, end, timeout=30):
    r = requests.get(url, headers={"Range": f"bytes={start}-{end-1}"}, timeout=timeout)
    if r.status_code not in [200, 206]:
        raise Exception(f"HTTP {r.status_code}")
    return r.content


def read_pixel_gcp(url, target_lat, target_lng):
    """Read pixel from GeoTIFF COG using GCP-based georeferencing"""
    fmt = "<"
    try:
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
        tile_off_ptr   = tags[324][2]; n_tiles_count = tags[324][1]
        tile_byte_ptr  = tags[325][2]

        # Read GCPs (tag 33922)
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

        cols = np.array([g[0] for g in gcps])
        rows = np.array([g[1] for g in gcps])
        lngs = np.array([g[2] for g in gcps])
        lats = np.array([g[3] for g in gcps])

        A = np.column_stack([lngs, lats, np.ones(len(gcps))])
        col_c, _, _, _ = lstsq(A, cols, rcond=None)
        row_c, _, _, _ = lstsq(A, rows, rcond=None)

        est_col = int(col_c[0]*target_lng + col_c[1]*target_lat + col_c[2])
        est_row = int(row_c[0]*target_lng + row_c[1]*target_lat + row_c[2])

        if not (0 <= est_col < width and 0 <= est_row < height):
            return None

        # Find and read tile
        tiles_across = (width + tile_w - 1) // tile_w
        tx = est_col // tile_w; ty = est_row // tile_h
        tile_idx = ty * tiles_across + tx

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

        lc = est_col % tile_w; lr = est_row % tile_h
        px_off = (lr * tile_w + lc) * 2
        if px_off + 2 > len(raw): return None

        dn = struct.unpack_from(f"{fmt}H", raw, px_off)[0]
        if dn == 0: return None

        sigma0 = (dn * dn) / (600 * 600)
        return round(10 * np.log10(sigma0 + 1e-10), 3)

    except Exception:
        return None


def get_sar_timeseries_polygon(polygon_coords, start_date, end_date,
                                client_id, client_secret, interval_days=12):
    """
    Extract SAR time series for a polygon using Element84 STAC + COG.
    Free — no CDSE processing units required.
    client_id and client_secret are unused (kept for API compatibility).
    """
    bbox = polygon_to_bbox(polygon_coords)
    lat_c = (bbox[1] + bbox[3]) / 2
    lng_c = (bbox[0] + bbox[2]) / 2

    # Search all S1 scenes in date range
    try:
        r = requests.post(
            f"{STAC_URL}/search",
            json={
                "collections": ["sentinel-1-grd"],
                "bbox": bbox,
                "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
                "limit": 200
            },
            timeout=30
        )
        if r.status_code != 200:
            return []
        features = r.json().get("features", [])
    except Exception:
        return []

    from datetime import datetime, timedelta
    observations = []
    scenes_to_fetch = []
    seen_dates = set()
    last_kept = None

    for f in sorted(features, key=lambda x: x["properties"].get("datetime","")):
        date_str = f["properties"].get("datetime", "")[:10]
        if not date_str: continue
        # Apply interval filter
        curr_date = datetime.strptime(date_str, "%Y-%m-%d")
        if last_kept and (curr_date - last_kept).days < interval_days:
            continue
        if date_str in seen_dates: continue
        seen_dates.add(date_str)
        last_kept = curr_date

        vh_url = s3_to_https(f["assets"].get("vh", {}).get("href", ""))
        vv_url = s3_to_https(f["assets"].get("vv", {}).get("href", ""))

        if not vh_url:
            observations.append({"date": date_str, "available": False})
            continue

        print(f"Fetching SAR polygon {date_str}...")

        scenes_to_fetch.append((date_str, vh_url, vv_url))

    # Parallel fetch
    from concurrent.futures import ThreadPoolExecutor
    def fetch_scene(args):
        date_str, vh_url, vv_url = args
        vh_db = read_pixel_gcp(vh_url, lat_c, lng_c)
        vv_db = read_pixel_gcp(vv_url, lat_c, lng_c) if vv_url else None
        if vh_db is not None:
            vh_lin = 10 ** (vh_db / 10)
            vv_lin = 10 ** (vv_db / 10) if vv_db else None
            rvi = round(min(1.0,(4*vh_lin)/((vv_lin or vh_lin)+vh_lin+1e-10)),4) if vv_lin else None
            return {"date":date_str,"available":True,"vh":vh_db,"vv":vv_db,"rvi":rvi}
        return {"date":date_str,"available":False}

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fetch_scene, scenes_to_fetch))
    observations.extend(results)
    return sorted(observations, key=lambda x: x["date"])
