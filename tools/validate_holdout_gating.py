import json, numpy as np, warnings
warnings.filterwarnings("ignore")
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

CONFIDENCE_THRESHOLD = 0.60

print("🏁 Starting Independent Validation Holdout Study...")
print(f"🔒 Confidence Gate Threshold Set to: {CONFIDENCE_THRESHOLD * 100:.1f}%\n")

with open("/workspaces/crop-trajectory/data/dataset_535.json") as f:
    data = json.load(f)

CM = {"Grassland":"Grassland","Barley":"Barley","Wheat":"Wheat","Oats":"Oats","Oilseed Rape":"Oilseed Rape","Maize":"Maize","Beans":"Beans"}
filtered_data = [d for d in data if d.get("label") in CM]
labels = [CM[d["label"]] for d in filtered_data]

def gm(dic, m):
    v = dic.get(str(m), dic.get(m, 0))
    return float(v) if v and float(v) != 0.0 else np.nan

all_ndvis, all_ndres, all_vhs, all_vvs = [{m: [] for m in range(1, 13)} for _ in range(4)]
for d in filtered_data:
    for m in range(1, 13):
        n_v  = gm(d.get("monthly_ndvi", {}), m)
        nr_v = gm(d.get("monthly_ndre", {}), m)
        vh_v = gm(d.get("monthly_vh", {}), m)
        vv_v = gm(d.get("monthly_vv", {}), m)
        if not np.isnan(n_v): all_ndvis[m].append(n_v)
        if not np.isnan(nr_v): all_ndres[m].append(nr_v)
        if not np.isnan(vh_v): all_vhs[m].append(vh_v)
        if not np.isnan(vv_v): all_vvs[m].append(vv_v)

default_ndvi = [np.median(all_ndvis[m]) if all_ndvis[m] else 0.4 for m in range(1, 13)]
default_ndre = [np.median(all_ndres[m]) if all_ndres[m] else 0.3 for m in range(1, 13)]
default_vh   = [np.median(all_vhs[m]) if all_vhs[m] else -17.0 for m in range(1, 13)]
default_vv   = [np.median(all_vvs[m]) if all_vvs[m] else -11.0 for m in range(1, 13)]

def interp(arr, fallback):
    x = np.arange(12)
    mask = ~np.isnan(arr)
    if mask.sum() >= 2: return np.interp(x, x[mask], arr[mask])
    elif mask.sum() == 1: return np.full(12, arr[mask])
    return np.array(fallback).copy()

X_full = []
for d in filtered_data:
    ndi = interp(np.array([gm(d.get("monthly_ndvi", {}), m) for m in range(1, 13)]), default_ndvi)
    nri = interp(np.array([gm(d.get("monthly_ndre", {}), m) for m in range(1, 13)]), default_ndre)
    vhi = interp(np.array([gm(d.get("monthly_vh", {}), m) for m in range(1, 13)]), default_vh)
    vvi = interp(np.array([gm(d.get("monthly_vv", {}), m) for m in range(1, 13)]), default_vv)
    area = float(d.get("area_ha", d.get("properties", {}).get("area_ha", 5.0)))
    perim = float(d.get("perimeter_m", d.get("properties", {}).get("perimeter_m", 400.0)))
    compactness = (4.0 * np.pi * area * 10000.0) / (perim ** 2 + 1e-6)
    elongation = perim / (4.0 * np.sqrt(area * 10000.0) + 1e-6)
    X_full.append(list(ndi) + list(nri) + list(vhi) + list(vvi) + [area, perim, compactness, elongation])

X_full = np.array(X_full, dtype=np.float64)
le = LabelEncoder(); y = le.fit_transform(labels)

X_train_full, X_val_full, y_train, y_val = train_test_split(X_full, y, test_size=0.20, stratify=y, random_state=42)

model_finder = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.06, loss_function="MultiClass", verbose=0, random_seed=42)
model_finder.fit(X_train_full, y_train)
opt_indices = np.argsort(model_finder.get_feature_importance())[::-1][:30]

X_train, X_val = X_train_full[:, opt_indices], X_val_full[:, opt_indices]
clf = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.06, loss_function="MultiClass", verbose=0, random_seed=42)
clf.fit(X_train, y_train)

raw_preds = clf.predict(X_val).flatten()
probs = clf.predict_proba(X_val)

gated_preds, gated_y_val = [], []
unknown_count = 0
total_samples = len(X_val)

print("📋 Sample Inferences (First 20 Unseen Parcels):")
print(f"  {'True Class':<15} | {'Predicted Class':<15} | {'Confidence':<10} | {'Status'} ")
print("  " + "-"*65)

for i in range(total_samples):
    true_lbl = le.inverse_transform([y_val[i]])[0]
    pred_idx = int(raw_preds[i])
    pred_lbl = le.inverse_transform([pred_idx])[0]
    conf = float(probs[i, pred_idx])
    if conf < CONFIDENCE_THRESHOLD:
        final_pred = "Unknown"
        unknown_count += 1
        status = "⚠️ REJECTED"
    else:
        final_pred = pred_lbl
        status = "✅ ACCEPTED"
        gated_preds.append(pred_idx)
        gated_y_val.append(y_val[i])
    if i < 20:
        print(f"  {true_lbl:<15} | {final_pred:<15} | {conf*100:.1f}%     | {status}")

raw_acc = np.mean(raw_preds == y_val) * 100
print(f"\n📈 Validation Performance Summary:")
print(f"  Raw Accuracy on Unseen Holdout: {raw_acc:.1f}%")
print(f"  Gated Parcel Rejection Rate   : {(unknown_count/total_samples)*100:.1f}% ({unknown_count}/{total_samples} fields)")

if gated_preds:
    gated_acc = np.mean(np.array(gated_preds) == np.array(gated_y_val)) * 100
    print(f"  Precision of Accepted Cargo   : {gated_acc:.1f}% 🔥")
    cm = confusion_matrix(gated_y_val, gated_preds)
    print("\n--- Gated Confusion Matrix (Accepted Only) ---")
    for idx, c_idx in enumerate(np.unique(gated_y_val)):
        c_name = le.inverse_transform([c_idx])[0]
        print(f"  {c_name:<15}: {cm[idx]}")
