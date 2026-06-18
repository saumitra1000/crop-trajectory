import json, numpy as np, warnings
warnings.filterwarnings("ignore")
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from collections import Counter

with open("/workspaces/crop-trajectory/data/dataset_535.json") as f:
    data = json.load(f)

CM = {"Grassland":"Grassland","Barley":"Barley","Wheat":"Wheat","Oats":"Oats","Oilseed Rape":"Oilseed Rape","Maize":"Maize","Beans":"Beans"}
data = [d for d in data if d.get("label") in CM]
labels = [CM[d["label"]] for d in data]

print("--- 7-Class Stratified Data Distribution ---")
print(Counter(labels))

def gm(dic, m):
    v = dic.get(str(m), dic.get(m, 0))
    return float(v) if v and float(v) != 0.0 else np.nan

all_ndvis, all_ndres, all_vhs, all_vvs = [{m: [] for m in range(1, 13)} for _ in range(4)]
for d in data:
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

feat_names = ([f"NDVI_M{m}" for m in range(1,13)] + [f"NDRE_M{m}" for m in range(1,13)] + [f"VH_M{m}" for m in range(1,13)] + [f"VV_M{m}" for m in range(1,13)] + ["Geo_Area", "Geo_Perimeter", "Geo_Compactness", "Geo_Elongation"])

X_list = []
for d in data:
    ndi = interp(np.array([gm(d.get("monthly_ndvi", {}), m) for m in range(1, 13)]), default_ndvi)
    nri = interp(np.array([gm(d.get("monthly_ndre", {}), m) for m in range(1, 13)]), default_ndre)
    vhi = interp(np.array([gm(d.get("monthly_vh", {}), m) for m in range(1, 13)]), default_vh)
    vvi = interp(np.array([gm(d.get("monthly_vv", {}), m) for m in range(1, 13)]), default_vv)
    area = float(d.get("area_ha", d.get("properties", {}).get("area_ha", 5.0)))
    perim = float(d.get("perimeter_m", d.get("properties", {}).get("perimeter_m", 400.0)))
    compactness = (4.0 * np.pi * area * 10000.0) / (perim ** 2 + 1e-6)
    elongation = perim / (4.0 * np.sqrt(area * 10000.0) + 1e-6)
    X_list.append(list(ndi) + list(nri) + list(vhi) + list(vvi) + [area, perim, compactness, elongation])

X = np.array(X_list, dtype=np.float64)
le = LabelEncoder(); y = le.fit_transform(labels)
print(f"Matrix Shape: {X.shape} | NaNs: {np.isnan(X).sum()}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.06, loss_function="MultiClass", verbose=0, random_seed=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print(f"\n🚀 CatBoost 7-Class Geometry Accuracy: {scores.mean()*100:.1f}% +/- {scores.std()*100:.1f}%")
print(f"Folds: {[round(s*100, 1) for s in scores]}")

model.fit(X, y)
importances = model.get_feature_importance()
sorted_indices = np.argsort(importances)[::-1]
print("\n--- Top 20 Most Predictive Features ---")
for i in range(min(20, len(feat_names))):
    idx = sorted_indices[i]
    print(f"  Rank {i+1:<2} | {feat_names[idx]:<16}: {importances[idx]:.2f}%")

yp = cross_val_predict(model, X, y, cv=cv)
cm = confusion_matrix(y, yp)
print("\n--- 7-Class CatBoost Confusion Matrix ---")
for i, c in enumerate(le.classes_):
    acc = round(cm[i,i]/max(cm[i].sum(),1)*100)
    leaks = " ".join(le.classes_[j][:4] + ":" + str(cm[i,j]) for j in range(len(le.classes_)) if j!=i and cm[i,j]>0)
    print(f"  {c:<15} {acc:>3}%  Leaks -> {leaks}")
