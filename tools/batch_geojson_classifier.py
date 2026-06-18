import json
import sys
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/workspaces/crop-trajectory')
from tools.inference_driver import predict_from_observations
from tools.consensus_engine import evaluate_crop_consensus

def calculate_polygon_perimeter(coords):
    try:
        pts = coords if len(coords) == 1 and isinstance(coords, list) else coords
        total_p = 0.0
        for i in range(len(pts)):
            p1, p2 = pts[i], pts[(i + 1) % len(pts)]
            total_p += np.sqrt((float(p2) - float(p1))**2 + (float(p2) - float(p1))**2) * 111320.0
        return max(total_p, 100.0)
    except:
        return 400.0

def execute_batch_geojson_classification(input_path, output_path):
    print("🎬 Running Production Multi-Tier GeoJSON Batch Processor...")
    if not os.path.exists(input_path):
        print(f"❌ Input file missing at: {input_path}")
        return
        
    with open(input_path, "r") as f:
        geojson_data = json.load(f)
        
    features = geojson_data.get("features", [])
    total_features = len(features)
    
    auto_accept_features = []
    review_queue_features = []
    t1, t2, t3 = 0, 0, 0
    
    for idx, feature in enumerate(features):
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        fid = props.get("parcel_id", f"BTI-{idx:03}")
        
        if geom.get("type") not in ["Polygon", "MultiPolygon"] or "coordinates" not in geom:
            continue
            
        poly_coords = geom["coordinates"]
        area_ha = float(props.get("area_ha", 4.5))
        perimeter_m = calculate_polygon_perimeter(poly_coords)
        
        res = predict_from_observations(
            poly_coords,
            client_id=os.environ.get("COP0_ID", "DUMMY_ID"),
            client_secret=os.environ.get("COP0_SECRET", "DUMMY_SECRET"),
            parcel_id=fid,
            sar_observations=None,
            area_ha=area_ha,
            perimeter_m=perimeter_m
        )
        
        pred_crop = res["predicted_crop"]
        confidence = res["confidence_pct"]
        tier = res["tier"]
        
        dafm_label = props.get("true_crop_class", props.get("crop_class", "Grassland"))
        sar_heuristic = "Grassland" if area_ha > 3.0 else pred_crop
        consensus_status = evaluate_crop_consensus(dafm_label, sar_heuristic, pred_crop)
        
        new_props = dict(props)
        new_props["raw_prediction"] = pred_crop
        new_props["inference_confidence"] = float(confidence)
        new_props["delivery_tier"] = tier
        new_props["automated_delivery"] = res["automated_delivery"]
        new_props["dafm_declaration"] = dafm_label
        new_props["sar_heuristic_prediction"] = sar_heuristic
        new_props["crop_consensus"] = consensus_status
        
        if tier == "Tier1":
            new_props["crop_prediction"] = pred_crop
            t1 += 1
            status_str = f"🚀 TIER 1 - AUTO-ACCEPT ({pred_crop} @ {confidence:.1f}%)"
            auto_accept_features.append({"type": "Feature", "geometry": geom, "properties": new_props})
        elif tier == "Tier2":
            new_props["crop_prediction"] = f"{pred_crop} (Low Confidence Hint)"
            t2 += 1
            status_str = f"⚠️ TIER 2 - REVIEW QUEUE ({pred_crop} @ {confidence:.1f}%)"
            review_queue_features.append({"type": "Feature", "geometry": geom, "properties": new_props})
        else:
            new_props["crop_prediction"] = "Unknown"
            t3 += 1
            status_str = f"🛑 TIER 3 - REJECTED ({confidence:.1f}%)"
            auto_accept_features.append({"type": "Feature", "geometry": geom, "properties": new_props})
            
        print(f"  Parcel ID: {fid:<12} | Model: {pred_crop:<10} | Confidence: {confidence:.1f}% | Routing: {tier}")
        
    with open(output_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": auto_accept_features}, f, indent=2)
        
    review_queue_path = "/workspaces/crop-trajectory/data/review_queue.geojson"
    with open(review_queue_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": review_queue_features}, f, indent=2)
        
    print(f"\n📊 Batch Processing Complete. Review Queue count: {t2} fields.")

if __name__ == "__main__":
    execute_batch_geojson_classification("/workspaces/crop-trajectory/data/real_parcel.json", "/workspaces/crop-trajectory/data/classified_parcels.geojson")
