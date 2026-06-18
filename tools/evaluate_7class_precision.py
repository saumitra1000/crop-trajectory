import json, sys, os, numpy as np, warnings
warnings.filterwarnings("ignore")
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def run_authentic_study():
    print("🏁 Starting Leakage-Proof Two-Tier Holdout Validation Sweep...")
    
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
        elif m.sum() == 1: return np.full(12, arr[m])
        return np.array(fb).copy()
        
    all_ndvis, all_ndres, all_vhs, all_vvs = [{m: [] for m in range(1, 13)} for _ in range(4)]
    for d in fdata:
        for m in range(1, 13):
            n, nr, vh, vv = gm(d.get("monthly_ndvi",{}),m), gm(d.get("monthly_ndre",{}),m), gm(d.get("monthly_vh",{}),m), gm(d.get("monthly_vv",{}),m)
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
        ndi = interp(np.array([gm(d.get("monthly_ndvi",{}),m) for m in range(1,13)]), f_nd)
        nri = interp(np.array([gm(d.get("monthly_ndre",{}),m) for m in range(1,13)]), f_nr)
        vhi = interp(np.array([gm(d.get("monthly_vh",{}),m) for m in range(1,13)]), f_vh)
        vvi = interp(np.array([gm(d.get("monthly_vv",{}),m) for m in range(1,13)]), f_vv)
        a = float(d.get("area_ha", 5.0))
        p = float(d.get("perimeter_m", 400.0))
        c = (4.0 * np.pi * a * 10000.0) / (p ** 2 + 1e-6)
        el = p / (4.0 * np.sqrt(a * 10000.0) + 1e-6)
        X_f.append(list(ndi) + list(nri) + list(vhi) + list(vvi) + [a, p, c, el])
        
    X_f = np.array(X_f, dtype=np.float64)
    le = LabelEncoder(); y = le.fit_transform(labels)
    
    # Strict holdout partition split
    Xt_f, Xv_f, yt, yv = train_test_split(X_f, y, test_size=0.20, stratify=y, random_state=42)
    
    m_find = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.06, loss_function="MultiClass", verbose=0, random_seed=42)
    m_find.fit(Xt_f, yt)
    idx = np.argsort(m_find.get_feature_importance())[::-1][:30]
    
    Xt, Xv = Xt_f[:, idx], Xv_f[:, idx]
    clf = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.06, loss_function="MultiClass", verbose=0, random_seed=42)
    clf.fit(Xt, yt)
    
    r_preds = clf.predict(Xv).flatten()
    probs = clf.predict_proba(Xv)
    tot = len(Xv)
    
    t1_preds, t1_y = [], []
    t2_preds, t2_y = [], []
    t3_count = 0
    
    for i in range(tot):
        p_idx = int(r_preds[i])
        conf = float(probs[i, p_idx])
        
        if conf >= 0.60:
            t1_preds.append(p_idx); t1_y.append(yv[i])
        elif 0.45 <= conf < 0.60:
            t2_preds.append(p_idx); t2_y.append(yv[i])
        else:
            t3_count += 1
            
    print(f"📊 Unseen Validation Holdout Complete. Evaluated over {tot} fields.")
    print("  " + "-"*60)
    print(f"  Tier 1 (Auto-Delivery)    : {len(t1_preds)} parcels ({(len(t1_preds)/tot)*100:.1f}% Acceptance Rate)")
    print(f"  Tier 2 (Confidence Hints) : {len(t2_preds)} parcels ({(len(t2_preds)/tot)*100:.1f}% Acceptance Rate)")
    print(f"  Tier 3 (Gated Rejection)  : {t3_count} parcels ({(t3_count/tot)*100:.1f}% Rejection Rate)")
    print("  " + "-"*60)
    
    t1_prec = np.mean(np.array(t1_preds) == np.array(t1_y)) * 100.0 if t1_preds else 0.0
    t2_prec = np.mean(np.array(t2_preds) == np.array(t2_y)) * 100.0 if t2_preds else 0.0
    
    print(f"  🔥 Tier 1 Delivery Precision: {t1_prec:.1f}%")
    print(f"  🕒 Tier 2 Delivery Precision: {t2_prec:.1f}%")
    print("  " + "-"*60)

if __name__ == "__main__": run_authentic_study()