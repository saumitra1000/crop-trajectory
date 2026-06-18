import json, sys, traceback, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/workspaces/crop-trajectory")
from tools.inference_driver import predict_live_lpis_parcel

def debug_and_test_lpis():
    print("--- Starting Un-Muted Diagnostic Test Suite ---")
    try:
        with open("/workspaces/crop-trajectory/data/dataset_535.json") as f: data = json.load(f)
    except Exception as e:
        print("❌ Failed to load file:", e); return

    if data and len(data) > 0:
        sample = data[0]
        print(f"\n[INSPECTION] Total Records: {len(data)}")
        print("[INSPECTION] First element key attributes:", list(sample.keys()))
    else:
        print("❌ Dataset is empty"); return

    geom_block = sample.get("geometry", sample)
    poly = geom_block.get("coordinates", [[[-8.91, 52.14], [-8.90, 52.14], [-8.90, 52.15], [-8.91, 52.15], [-8.91, 52.14]]])
    if "coordinates" not in geom_block and "coordinates" in sample: poly = sample["coordinates"]

    print(f"\n[INSPECTION] Target poly Python type: {type(poly)}")
    print("[INSPECTION] Target poly values:", str(poly)[:120])

    print("\n🛰️ Triggering live pipeline extraction inference step...")
    try:
        area = float(sample.get("area_ha", 5.0))
        perim = float(sample.get("perimeter_m", 400.0))
        pred_crop, confidence = predict_live_lpis_parcel(poly, area_ha=area, perimeter_m=perim)
        print(f"Prediction: {pred_crop} | Confidence: {confidence*100:.1f}%")
    except Exception as e:
        print("\n💥 EXCEPTION CAUGHT! Printing full raw traceback loop:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_and_test_lpis()