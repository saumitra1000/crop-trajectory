import json, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/workspaces/crop-trajectory")
from tools.inference_driver import predict_live_lpis_parcel

def execute_live_lpis_test():
    print("--- Starting End-to-End Live LPIS Validation Test ---")
    
    real_lpis_polygon = [
        [-7.915, 53.342],
        [-7.910, 53.342],
        [-7.910, 53.347],
        [-7.915, 53.347],
        [-7.915, 53.342]
    ]
    
    area_ha = 3.8
    perimeter_m = 320.0
    
    print("📌 Injecting real spatial geometry to extractor...")
    print("   Field Dimensions: Area = " + str(area_ha) + " Ha | Perimeter = " + str(perimeter_m) + " m")
    
    pred_crop, confidence = predict_live_lpis_parcel(real_lpis_polygon, area_ha=area_ha, perimeter_m=perimeter_m)
    
    print("")
    print("--- LIVE INFERENCE DEPLOYMENT SUMMARY ---")
    print("  Gated Model Classification Output: " + str(pred_crop))
    print("  Model Certainty Confidence Metric: " + str(round(confidence * 100, 1)) + "%")
    print("-------------------------------------------")
    if str(pred_crop) == "Unknown":
        print("⚠️ Result: Gated successfully. Input dataset generated high uncertainty or cloud gaps.")
    else:
        print("✅ Result: Cleared confidence gate! Output asset ready for transmission.")

if __name__ == "__main__":
    execute_live_lpis_test()