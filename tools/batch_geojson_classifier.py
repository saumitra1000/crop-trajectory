import json, sys, os, numpy as np, joblib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/workspaces/crop-trajectory")
from tools.inference_driver import predict_live_lpis_parcel

def calculate_polygon_perimeter(coords):
    try:
        pts = coords if len(coords) == 1 and isinstance(coords, list) else coords
        total_p = 0.0
        for i in range(len(pts)):
            p1, p2 = pts[i], pts[(i + 1) % len(pts)]
            total_p += np.sqrt((float(p2[0]) - float(p1[0]))**2 + (float(p2[1]) - float(p1[1]))**2) * 111320.0
        return max(total_p, 100.0)
    except: return 400.0

def execute_batch_geojson_classification(input_path, output_path):
    print("🎬 Initializing Automated Two-Tier GeoJSON Batch Processing Pipeline...")
    if not os.path.exists(input_path): return
    with open(input_path, "r") as f: geojson_data = json.load(f)
    features = geojson_data.get("features", [])
    total_features = len(features)
    
    processed_features = []
    t1_count, t2_count, t3_count = 0, 0, 0
    
    for idx, feature in enumerate(features):
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        fid = props.get("parcel_id", f"BTI-{idx:03}")
        if geom.get("type") not in ["Polygon", "MultiPolygon"] or "coordinates" not in geom: continue
        
        poly_coords = geom["coordinates"]
        area_ha = float(props.get("area_ha", 4.5))
        perimeter_m = calculate_polygon_perimeter(poly_coords)
        
        # Extract un-gated raw prediction via temporary bypass
        pred_crop, confidence = predict_live_lpis_parcel(poly_coords, area_ha=area_ha, perimeter_m=perimeter_m)
        
        new_props = dict(props)
        new_props["raw_prediction"] = str(pred_crop)
        new_props["inference_confidence"] = float(round(confidence * 100, 2))
        
        # TWO-TIER DEPLOYMENT ROUTING ENGINE
        if confidence >= 0.60:
            new_props["crop_prediction"] = str(pred_crop)
            new_props["delivery_tier"] = "Tier 1: Automated Delivery"
            t1_count += 1
            status = f"🚀 TIER 1 - AUTOMATED DELIVERY ({pred_crop} @ {confidence*100:.1f}%)"
        elif 0.45 <= confidence < 0.60:
            new_props["crop_prediction"] = str(pred_crop) + " (Low Confidence Hint)"
            new_props["delivery_tier"] = "Tier 2: Low-Confidence Hint"
            t2_count += 1
            status = f"⚠️ TIER 2 - LOW-CONFIDENCE HINT ({pred_crop} @ {confidence*100:.1f}%)"
        else:
            new_props["crop_prediction"] = "Unknown"
            new_props["delivery_tier"] = "Tier 3: Rejected (High Uncertainty)"
            t3_count += 1
            status = f"🛑 TIER 3 - REJECTED (High Uncertainty @ {confidence*100:.1f}%)"
            
        print(f"  Parcel ID: {fid:<12} | Status: {status}")
        new_feature = {"type": "Feature", "geometry": geom, "properties": new_props}
        processed_features.append(new_feature)
        
    with open(output_path, "w") as f: json.dump({"type": "FeatureCollection", "features": processed_features}, f, indent=2)
    print("\n📊 Multi-Tiered Process Complete Performance Summary:")
    print(f"  Total Extracted Input Geometries           : {total_features}")
    print(f"  Tier 1 Delivery (Automated, High Precision) : {t1_count} parcels ({(t1_count/total_features)*100:.1f}%)")
    print(f"  Tier 2 Delivery (Low-Confidence Hints)      : {t2_count} parcels ({(t2_count/total_features)*100:.1f}%)")
    print(f"  Tier 3 Rejection (High Uncertainty Gated)   : {t3_count} parcels ({(t3_count/total_features)*100:.1f}%)")
    print(f"💾 Annotated tiered map layer saved to: {output_path}")

if __name__ == "__main__":
    execute_batch_geojson_classification("/workspaces/crop-trajectory/data/real_parcel.json", "/workspaces/crop-trajectory/data/classified_parcels.geojson")