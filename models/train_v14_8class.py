import json, numpy as np, warnings
warnings.filterwarnings("ignore")
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from collections import Counter

with open("/workspaces/crop-trajectory/data/dataset_535.json") as f:
    data = json.load(f)

labels = [d["label"] for d in data]
print("Classes:", Counter(labels))

def gm(dic, m):
    v = dic.get(str(m), dic.get(m, 0))
    return float(v) if v and float(v) != 0.0 else np.nan

# 1. Compute robust baseline medians to handle sparse optical gaps
all_ndvis, all_ndres, all_vhs, all_vvs = [{m: [] for m in range(1, 13)} for _ in range(4)]
for d in data:
    for m in range(1, 13):
        n_v = gm(d.get("monthly_ndvi", {}), m)
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
    if mask.sum() >= 2:
        return np.interp(x, x[mask], arr[mask])
    elif mask.sum() == 1:
        return np.full(12, arr[mask][0])
    return np.array(fallback).copy()

X_list = []
for d in data:
    # 2. Extract and interpolate all 4 core sensor domains natively
    ndi = interp(np.array([gm(d.get("monthly_ndvi", {}), m) for m in range(1, 13)]), default_ndvi)
    nri = interp(np.array([gm(d.get("monthly_ndre", {}), m) for m in range(1, 13)]), default_ndre)
    vhi = interp(np.array([gm(d.get("monthly_vh", {}), m) for m in range(1, 13)]), default_vh)
    vvi = interp(np.array([gm(d.get("monthly_vv", {}), m) for m in range(1, 13)]), default_vv)

    db_diff = vvi - vhi

    # 3. Inject 8 targeted structural and phenology shape descriptors
    descriptors = [
        float(np.max(ndi)),                             # Peak NDVI
        float(np.max(ndi) - np.min(ndi)),               # Dynamic NDVI Range
        float(np.argmax(ndi) + 1),                      # Peak NDVI Month
        float(np.nanargmax(np.diff(ndi)) + 1),          # Growth Velocity Anchor
        float(np.mean(db_diff)),                        # Baseline Canopy Geometry
        float(np.std(vhi)),                             # Annual Radar Texture Variance
        float(vhi[5] - vhi[7]),                         # Late-Summer Canopy Drop (June vs August)
        float(ndi[5] - ndi[7])                          # June to August Harvest Step
    ]

    # Build the full 56-dimensional feature space
    X_list.append(list(ndi) + list(nri) + list(vhi) + list(vvi) + descriptors)

X = np.array(X_list, dtype=np.float64)
print(f"Matrix Shape: {X.shape} | Remaining NaNs: {np.isnan(X).sum()}")

le = LabelEncoder()
y  = le.fit_transform(labels)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in [
    ("BRF", BalancedRandomForestClassifier(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)),
    ("XGB", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=42, n_jobs=-1))
]:
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"\n{name} Full-Sequence v14 Accuracy: {scores.mean()*100:.1f}% +/- {scores.std()*100:.1f}%")
    yp = cross_val_predict(clf, X, y, cv=cv)
    cm = confusion_matrix(y, yp)
    print("--- Confusion Matrix ---")
    for i, c in enumerate(le.classes_):
        acc = round(cm[i, i] / max(cm[i].sum(), 1) * 100)
        leaks = " ".join(f"{le.classes_[j][:4]}:{cm[i,j]}" for j in range(len(le.classes_)) if j != i and cm[i, j] > 0)
        print(f"  {c:<15} {acc:>3}%  {leaks}")
