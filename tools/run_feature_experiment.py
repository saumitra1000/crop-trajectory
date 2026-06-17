import json, numpy as np, warnings
warnings.filterwarnings("ignore")
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

with open("/workspaces/crop-trajectory/data/dataset_535.json") as f:
    data = json.load(f)

CM = {"Grassland":"Grassland","Barley":"Barley","Wheat":"Wheat","Oats":"Oats","Oilseed Rape":"Oilseed Rape","Maize":"Maize","Beans":"Beans"}
data = [d for d in data if d.get("label") in CM]
labels = [CM[d["label"]] for d in data]

def gm(dic, m):
    v = dic.get(str(m), dic.get(m, 0))
    return float(v) if v and float(v) != 0.0 else np.nan

all_ndvis, all_ndres, all_vhs, all_vvs = [{m: [] for m in range(1, 13)} for _ in range(4)]
for d in data:
    for m in range(1, 13):
        n_v, nr_v, vh_v, vv_v = gm(d.get("monthly_ndvi",{}),m), gm(d.get("monthly_ndre",{}),m), gm(d.get("monthly_vh",{}),m), gm(d.get("monthly_vv",{}),m)
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
for d in data:
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

model_full = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.06, loss_function="MultiClass", verbose=0, random_seed=42)
model_full.fit(X_full, y)
importances = model_full.get_feature_importance()
sorted_indices = np.argsort(importances)[::-1]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("--- Controlled Feature Density Experiment Results ---")

for k in [10, 20, 30, 52]:
    target_indices = sorted_indices[:k]
    X_subset = X_full[:, target_indices]
    model = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.06, loss_function="MultiClass", verbose=0, random_seed=42)
    scores = cross_val_score(model, X_subset, y, cv=cv, scoring="accuracy")
    print(f"Top {k:<2} Features | Accuracy: {scores.mean()*100:.1f}% +/- {scores.std()*100:.1f}%")
