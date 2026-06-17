import json
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

def run_end_to_end_test():
    print("🎬 Starting End-to-End Inference Pipeline Test...")
    
    # 1. Load the serialized production assets
    try:
        model = joblib.load("/workspaces/crop-trajectory/models/production_catboost_7class.pkl")
        le = joblib.load("/workspaces/crop-trajectory/models/encoder_7class.pkl")
        opt_indices = joblib.load("/workspaces/crop-trajectory/models/optimal_indices.pkl")
        print("✅ Step 1/4: Production model components loaded successfully from disk.")
    except Exception as e:
        print(f"❌ Step 1/4 Error: Failed to load models. {e}")
        return

    # 2. Extract a real test record from the dataset
    try:
        with open('/workspaces/crop-trajectory/data/dataset_535.json') as f:
            data = json.load(f)
        
        CM = {"Grassland":"Grassland","Barley":"Barley","Wheat":"Wheat","Oats":"Oats","Oilseed Rape":"Oilseed Rape","Maize":"Maize","Beans":"Beans"}
        test_sample = None
        for d in data:
            if d.get('label') in CM:
                test_sample = d
                break
                
        if test_sample is None:
            print("❌ Step 2/4 Error: No valid 7-class samples found in dataset.")
            return
            
        print(f"✅ Step 2/4: Mocking satellite COG stream for Parcel ID: {test_sample.get('id', 'UKN-404')} (True Class: {test_sample['label']})")
    except Exception as e:
        print(f"❌ Step 2/4 Error: {e}")
        return

    # 3. Simulate the featurizer & pure-NumPy interpolation layer
    def gm(dic, m):
        v = dic.get(str(m), dic.get(m, 0))
        return float(v) if v and float(v) != 0.0 else np.nan

    nd_raw = np.array([gm(test_sample.get('monthly_ndvi', {}), m) for m in range(1, 13)])
    nr_raw = np.array([gm(test_sample.get('monthly_ndre', {}), m) for m in range(1, 13)])
    vh_raw = np.array([gm(test_sample.get('monthly_vh', {}), m) for m in range(1, 13)])
    vv_raw = np.array([gm(test_sample.get('monthly_vv', {}), m) for m in range(1, 13)])

    fallback_nd = [0.4]*12
    fallback_nr = [0.3]*12
    fallback_vh = [-17.0]*12
    fallback_vv = [-11.0]*12

    def interp(arr, fallback):
        x = np.arange(12)
        mask = ~np.isnan(arr)
        if mask.sum() >= 2: return np.interp(x, x[mask], arr[mask])
        elif mask.sum() == 1: return np.full(12, arr[mask])
        return np.array(fallback).copy()

    ndi = interp(nd_raw, fallback_nd)
    nri = interp(nr_raw, fallback_nr)
    vhi = interp(vh_raw, fallback_vh)
    vvi = interp(vv_raw, fallback_vv)

    area = float(test_sample.get('area_ha', 5.0))
    perim = float(test_sample.get('perimeter_m', 400.0))
    compactness = (4.0 * np.pi * area * 10000.0) / (perim ** 2 + 1e-6)
    elongation = perim / (4.0 * np.sqrt(area * 10000.0) + 1e-6)

    X_full = np.array(list(ndi) + list(nri) + list(vhi) + list(vvi) + [area, perim, compactness, elongation]).reshape(1, -1)
    print(f"✅ Step 3/4: Feature array generated. Shape: {X_full.shape} | NaNs: {np.isnan(X_full).sum()}")

    # 4. Mask down to Top-30 and run live classification inference
    X_opt = X_full[:, opt_indices]
    print(f"📌 Masking complete. Input shape reduced from 52 down to optimized: {X_opt.shape}")

    pred_idx = model.predict(X_opt).flatten()
    probs = model.predict_proba(X_opt)
    predicted_label = le.inverse_transform(pred_idx)

    # FIX: Explicit 2D matrix array index slicing to output clean string scalar conversions
    confidence = float(probs[0, pred_idx[0]])

    print("\n--- 🔮 LIVE INFERENCE RESULTS ---")
    print(f"  Predicted Crop Category : **{predicted_label[0]}**")
    print(f"  True Field Category      : **{test_sample['label']}**")
    print(f"  Classification Confidence: {confidence * 100:.1f}%")
    print("---------------------------------")
    print("🎉 Test execution finished with zero array drops!")

if __name__ == "__main__":
    run_end_to_end_test()
