import json
import sys
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

def execute_tier2_audit():
    print("🔬 Starting Raw Tier 2 Calibration & Unmasked Matrix Audit...")
    
    with open("/workspaces/crop-trajectory/data/dataset_535.json") as f:
        data = json.load(f)
        
    CM = {"Grassland":"Grassland","Barley":"Barley","Wheat":"Wheat","Oats":"Oats","Oilseed Rape":"Oilseed Rape","Maize":"Maize","Beans":"Beans"}
    fdata = [d for d in data if d.get("label") in CM]
    labels = [CM[d["label"]] for d in fdata]
    
    def gm(dic, m):
        if not dic or not isinstance(dic, dict): return np.nan
        v = dic.get(str(m), dic.get(m, 0))
        return float(v) if v and float(v) != 0.0 else np.nan
        
    def interp(arr, fb):
        x = np.arange(12); m = ~np.isnan(arr)
        if m.sum() >= 2: return np.interp(x, x[m], arr[m])
        elif m.sum() == 1: return np.full(12, arr[m][0])
        return np.array(fb).copy()
        
    all_ndvis, all_ndres, all_vhs, all_vvs = [{m: [] for m in range(1, 13)} for _ in range(4)]
    for d in fdata:
        for m in range(1, 13):
            n = gm(d.get("monthly_ndvi",{}), m)
            nr = gm(d.get("monthly_ndre",{}), m)
            vh = gm(d.get("monthly_vh",{}), m)
            vv = gm(d.get("monthly_vv",{}), m)
            if not np.isnan(n): all_ndvis[m].append(n)
            if not np.isnan(nr): all_ndres[m].append(nr)
            if not np.isnan(vh): all_vhs[m].append(vh)
            if not np.isnan(vv): all_vvs[m].append(vv)
            
    f_nd = [np.median(all_ndvis[m]) if all_ndvis[m] else 0.4 for m in range(1, 13)]
    f_nr = [np.median(all_ndres[m]) if all_ndres[m] else 0.3 for m in range(1, 13)]
    f_vh = [np.median(all_vhs[m]) if all_vhs[m] else -17.0 for m in range(1, 13)]
    f_vv = [np.median(all_vvs[m]) if all_vvs[m] else -11.0 for m in range(1, 13)]
    
    X_f = []
    for d in fdata:
        ndi = interp(np.array([gm(d.get("monthly_ndvi", {}), m) for m in range(1, 13)]), f_nd)
        nri = interp(np.array([gm(d.get("monthly_ndre", {}), m) for m in range(1, 13)]), f_nr)
        vhi = interp(np.array([gm(d.get("monthly_vh", {}), m) for m in range(1, 13)]), f_vh)
        vvi = interp(np.array([gm(d.get("monthly_vv", {}), m) for m in range(1, 13)]), f_vv)
        a = float(d.get("area_ha", 5.0))
        p = float(d.get("perimeter_m", 400.0))
        c = (4.0 * np.pi * a * 10000.0) / (p ** 2 + 1e-6)
        el = p / (4.0 * np.sqrt(a * 10000.0) + 1e-6)
        X_f.append(list(ndi) + list(nri) + list(vhi) + list(vvi) + [a, p, c, el])
        
    X_f = np.array(X_f, dtype=np.float64)
    le = LabelEncoder(); y = le.fit_transform(labels)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    t2_true, t2_pred, grassland_t2_scores = [], [], []
    
    print("🔄 Running 5-fold stratified cross-validation splits...")
    for train_idx, val_idx in cv.split(X_f, y):
        Xt_full, Xv_full = X_f[train_idx], X_f[val_idx]
        yt, yv = y[train_idx], y[val_idx]
        
        m_find = CatBoostClassifier(iterations=250, depth=5, learning_rate=0.06, loss_function="MultiClass", verbose=0, random_seed=42)
        m_find.fit(Xt_full, yt)
        idx = np.argsort(m_find.get_feature_importance())[::-1][:30]
        
        Xt, Xv = Xt_full[:, idx], Xv_full[:, idx]
        clf = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.06, loss_function="MultiClass", verbose=0, random_seed=42)
        clf.fit(Xt, yt)
        
        fold_preds = clf.predict(Xv).flatten()
        probs = clf.predict_proba(Xv)
        
        for i in range(len(val_idx)):
            p_idx = int(fold_preds[i])
            conf = float(probs[i, p_idx])
            true_idx = int(yv[i])
            
            # Isolate Tier 2 Fence (45% <= Confidence < 60%)
            if 0.45 <= conf < 0.60:
                t2_true.append(true_idx)
                t2_pred.append(p_idx)
                if true_idx == int(le.transform(["Grassland"])[0]):
                    grassland_t2_scores.append(conf)
                    
    print(f"\n📊 Tier 2 Distribution Audit Summary (45%-60% Confidence Interval):")
    print(f"  Total Unmasked Tier 2 Parcels Found: {len(t2_true)}")
    
    t2_true = np.array(t2_true); t2_pred = np.array(t2_pred)
    if len(t2_true) > 0:
        t2_acc = np.mean(t2_true == t2_pred) * 100.0
        print(f"  Standalone Precision of Tier 2 Fence: {t2_acc:.1f}%")
        
        print("\n--- 🛑 UNMASKED TIER 2 CONFUSION MATRIX ---")
        cm = confusion_matrix(t2_true, t2_pred, labels=range(len(le.classes_)))
        print(f"  {'True Class':<15} | Leaks By Predicted Index (Classes: {list(le.classes_)})")
        print("  " + "-" * 75)
        for i, c in enumerate(le.classes_):
            print(f"  {c:<15} | {cm[i].tolist()}")
    else:
        print("  No samples landed in the Tier 2 bracket.")
        
    if grassland_t2_scores:
        print("\n📈 Grassland Calibration Metrics:")
        print(f"  True Grasslands trapped inside Tier 2 : {len(grassland_t2_scores)}")
        print(f"  Median confidence of trapped pastures  : {np.median(grassland_t2_scores)*100:.1f}%")
        print("💡 Recalibration Target: Shifting gate to 50% safely absorbs these records.")

if __name__ == "__main__":
    execute_tier2_audit()
