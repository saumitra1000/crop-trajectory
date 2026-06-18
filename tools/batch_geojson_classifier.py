import json
import sys
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/workspaces/crop-trajectory')
from tools.inference_driver import predict_live_lpis_parcel

def calculate_polygon_perimeter(coords):
    """Calculates basic planar boundary perimeter lengths from coordinate outer rings"""
    try:
        pts = coords[0] if len(coords) == 1 and isinstance(coords[0], list) else coords
        total_p = 0.0
        for i in range(len(pts)):
            p1, p2 = pts[i], pts[(i + 1) % len(pts)]
            total_p += np.sqrt((float(p2[0]) - float(p1[0]))**2 + (float(p2[1]) - float(p1[1]))**2) * 111320.0
        return max(total_p, 100.0)
    except:
        return 400.0

def execute_batch_geojson_classification(input_path, output_path):
    print("🎬 Running Production Two-Tier GeoJSON Batch Processor...")
    if not os.path.exists(input_path):
        print(f"❌ Input file missing at: {input_path}")
        return
        
    with open(input_path, "r") as f:
        geojson_data = json.load(f)
        
    features = geojson_data.get("features", [])
    total_features = len(features)
    print(f"📌 Layer Parsing Successful. Found {total_features} features to classify.")
    
    processed_features = []
    t1_count, t2_count, t3_count = 0, 0, 0
    
    for idx, feature in enumerate(features):
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        fid = props.get("parcel_id", props.get("PARCEL_ID", f"BTI-{idx:03}"))
        
        if geom.get("type") not in ["Polygon", "MultiPolygon"] or "coordinates" not in geom:
            continue
            
        poly_coords = geom["coordinates"]
        area_ha = float(props.get("area_ha", props.get("AREA_HA", 4.5)))
        perimeter_m = calculate_polygon_perimeter(poly_coords)
        
        pred_crop, confidence = predict_live_lpis_parcel(poly_coords, area_ha=area_ha, perimeter_m=perimeter_m)
        pred_crop_clean = str(pred_crop).replace("[", "").replace("]", "").replace("'", "")
        
        new_props = dict(props)
        new_props["raw_prediction"] = pred_crop_clean
        new_props["inference_confidence"] = float(round(confidence * 100, 2))
        
        # TWO-TIER DEPLOYMENT ROUTING ENGINE (Validated fixed 60% Gate)
        if confidence >= 0.60:
            new_props["crop_prediction"] = pred_crop_clean
            new_props["delivery_tier"] = "Tier 1: Automated Delivery"
            t1_count += 1
            status_str = f"🚀 TIER 1 ({pred_crop_clean} @ {confidence*100:.1f}%)"
        elif 0.45 <= confidence < 0.60:
            new_props["crop_prediction"] = f"{pred_crop_clean} (Low Confidence Hint)"
            new_props["delivery_tier"] = "Tier 2: Low-Confidence Hint"
            t2_count += 1
            status_str = f"⚠️ TIER 2 ({pred_crop_clean} @ {confidence*100:.1f}%)"
        else:
            new_props["crop_prediction"] = "Unknown"
            new_props["delivery_tier"] = "Tier 3: Rejected (High Uncertainty)"
            t3_count += 1
            status_str = f"🛑 TIER 3 (Gated @ {confidence*100:.1f}%)"
            
        print(f"  [{idx+1}/{total_features}] Parcel ID: {fid:<12} | Status: {status_str}")
        new_feature = {"type": "Feature", "geometry": geom, "properties": new_props}
        processed_features.append(new_feature)
        
    with open(output_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": processed_features}, f, indent=2)
        
    print("\n📊 Multi-Tiered Process Complete Performance Summary:")
    print(f"  Total Extracted Geometries      : {total_features}")
    print(f"  Tier 1 Delivery (Auto-Pass)     : {t1_count} parcels")
    print(f"  Tier 2 Delivery (Unmasked Hint) : {t2_count} parcels")
    print(f"  Tier 3 Rejection (Gated Block)  : {t3_count} parcels")
    print(f"💾 Annotated tiered map layer saved to: {output_path}")

if __name__ == "__main__":
    execute_batch_geojson_classification("/workspaces/crop-trajectory/data/real_parcel.json", "/workspaces/crop-trajectory/data/classified_parcels.geojson")
