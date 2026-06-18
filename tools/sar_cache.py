import json
import os

import os as _os
CACHE_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "sar_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(parcel_id, start_date):
    """Generates a clean tracking path based on parcel ID and season start year"""
    # Extract only the year characters from the season date block string anchor
    year_anchor = str(start_date).split("-")[0]
    safe_id = "".join([c if c.isalnum() else "_" for c in str(parcel_id)])
    return os.path.join(CACHE_DIR, f"sar_{safe_id}_{year_anchor}.json")

def read_sar_cache(parcel_id, start_date):
    """Looks up and loads cached multi-temporal feature maps from disk"""
    target = get_cache_path(parcel_id, start_date)
    if os.path.exists(target):
        try:
            with open(target, "r") as f:
                return json.load(f), True
        except Exception:
            return None, False
    return None, False

def write_sar_cache(parcel_id, start_date, data_dict):
    """Persists extracted STAC/SAR sensor feature matrices to local disk storage"""
    if not data_dict or not isinstance(data_dict, (dict, list)):
        return False
    target = get_cache_path(parcel_id, start_date)
    try:
        with open(target, "w") as f:
            json.dump(data_dict, f, indent=2)
        return True
    except Exception:
        return False
