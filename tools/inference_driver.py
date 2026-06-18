import json
import os
import sys
import numpy as np
import pyproj
import rasterio
import math
import joblib
import warnings
warnings.filterwarnings('ignore')
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extractors.fusion_extractor import extract_fusion_features

VALID_SCL = {4, 5, 7}

def normalize_to_coordinate_ring(geom_input):
    """
    Safely unwraps nested GeoJSON polygon coordinate depths down to a flat
    sequence list of coordinate coordinate rings [[lng, lat], ...] iteratively.
    """
    if not isinstance(geom_input, list) or len(geom_input) == 0:
        return []
    
    current = geom_input
    for _ in range(5):
        if not isinstance(current, list) or len(current) == 0:
            return []
        first = current[0]
        if isinstance(first, (int, float)):
            return current
        if isinstance(first, list):
            if len(first) > 0 and isinstance(first[0], (int, float)):
                return current
            current = first
        else:
            break
    return current if isinstance(current, list) else []

def generate_interior_points(polygon):
    clean_pts = normalize_to_coordinate_ring(polygon)
    lngs = [float(pt[0]) for pt in clean_pts if isinstance(pt, (list, tuple)) and len(pt) >= 2]
    lats = [float(pt[1]) for pt in clean_pts if isinstance(pt, (list, tuple)) and len(pt) >= 2]
    if not lngs or not lats:
        return [(52.14, -8.91), (52.14, -8.90), (52.15, -8.91)]
    c_lng, c_lat = np.mean(lngs), np.mean(lats)
    offset_deg = 20.0 / 111320.0
    lat_scale = math.cos(math.radians(c_lat))
    return [(c_lat, c_lng), (c_lat, c_lng + (offset_deg / lat_scale)), (c_lat + offset_deg, c_lng)]

def predict_from_observations(polygon_geometry, client_id, client_secret, sar_observations=None, area_ha=5.0, perimeter_m=400.0):
    """
    🎯 High-Performance Inference wrapper that reuses pre-computed SAR vectors
    to completely eliminate duplicate network loops over Copernicus CDSE STAC servers.
    """
    try:
        import os
        _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model = joblib.load(os.path.join(_base, "models", "production_catboost_7class.pkl"))
        le = joblib.load(os.path.join(_base, "models", "encoder_7class.pkl"))
        opt_indices = joblib.load(os.path.join(_base, "models", "optimal_indices.pkl"))
    except Exception as e:
        print("❌ Model load error:", e)
        return "Unknown", 0.0

    clean_geom = normalize_to_coordinate_ring(polygon_geometry)
    if not clean_geom:
        return "Unknown", 0.0
        
    try:
        feat_dict = extract_fusion_features(
            clean_geom, 
            client_id, 
            client_secret,
            sar_observations=sar_observations,
            start_date="2024-10-01",
            end_date="2025-09-30"
        )
        if not feat_dict or not isinstance(feat_dict, dict):
            return "Unknown", 0.0
    except Exception as e:
        return "Unknown", 0.0

    def gm(dic, m):
        if not dic or not isinstance(dic, dict): return np.nan
        v = dic.get(str(m), dic.get(m, 0))
        return float(v) if v and float(v) != 0.0 else np.nan

    def interp_channel(arr, fallback):
        x = np.arange(12)
        mask = ~np.isnan(arr)
        if mask.sum() >= 2: return np.interp(x, x[mask], arr[mask])
        elif mask.sum() == 1: return np.full(12, arr[mask])
        return np.array(fallback).copy()

    ndi = interp_channel(np.array([gm(feat_dict.get("monthly_ndvi", {}), m) for m in range(1, 13)]), [0.4]*12)
    nri = interp_channel(np.array([gm(feat_dict.get("monthly_ndre", {}), m) for m in range(1, 13)]), [0.3]*12)
    vhi = interp_channel(np.array([gm(feat_dict.get("monthly_vh", {}), m) for m in range(1, 13)]), [-17.0]*12)
    vvi = interp_channel(np.array([gm(feat_dict.get("monthly_vv", {}), m) for m in range(1, 13)]), [-11.0]*12)

    compactness = (4.0 * np.pi * area_ha * 10000.0) / (perimeter_m ** 2 + 1e-6)
    elongation = perimeter_m / (4.0 * np.sqrt(area_ha * 10000.0) + 1e-6)
    geom_features = [area_ha, perimeter_m, compactness, elongation]

    X_full = np.array(list(ndi) + list(nri) + list(vhi) + list(vvi) + geom_features).reshape(1, -1)
    X_opt = X_full[:, opt_indices]

    raw_pred = model.predict(X_opt)
    
    # Safe Unpacking layer accommodating both native ndarray outputs and mock structures
    if isinstance(raw_pred, np.ndarray):
        pred_idx = int(raw_pred.ravel()[0])
    else:
        try:
            pred_idx = int(raw_pred)
        except:
            pred_idx = 0
        
    probs = model.predict_proba(X_opt)
    confidence = float(probs[0, pred_idx])
    predicted_crop = le.inverse_transform([pred_idx])

    if confidence < 0.60:
        return "Unknown", confidence
    
    crop_str = predicted_crop[0] if isinstance(predicted_crop, (list, np.ndarray)) else str(predicted_crop)
    return crop_str, confidence

def predict_live_lpis_parcel(polygon_geometry, area_ha=5.0, perimeter_m=400.0):
    return predict_from_observations(
        polygon_geometry, 
        client_id=os.environ.get("COP0_ID", "DUMMY_ID"), 
        client_secret=os.environ.get("COP0_SECRET", "DUMMY_SECRET"), 
        sar_observations=None, 
        area_ha=area_ha, 
        perimeter_m=perimeter_m
    )
