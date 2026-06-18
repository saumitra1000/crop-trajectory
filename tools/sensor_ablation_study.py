import json, numpy as np, warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load data asset pool
with open('/workspaces/crop-trajectory/data/dataset_535.json') as f:
    data = json.load(f)

CM = {'Grassland':'Grassland','Barley':'Barley','Wheat':'Wheat','Oats':'Oats','Oilseed Rape':'Oilseed Rape','Maize':'Maize','Beans':'Beans'}
data = [d for d in data if d.get('label') in CM]
labels = [CM[d['label']] for d in data]

def gm(dic, m):
    v = dic.get(str(m), dic.get(m, 0))
    return float(v) if v and float(v) != 0.0 else np.nan

def interp(arr, fallback):
    x = np.arange(12)
    mask = ~np.isnan(arr)
    if mask.sum() >= 2: return np.interp(x, x[mask], arr[mask])
    elif mask.sum() == 1: return np.full(12, arr[mask][0])
    return np.array(fallback).copy()

# Pre-calculate baseline defaults
all_ndvis, all_ndres, all_vhs, all_vvs = [{m: [] for m in range(1, 13)} for _ in range(4)]
for d in data:
    for m in range(1, 13):
        n_v, nr_v, vh_v, vv_v = gm(d.get('monthly_ndvi',{}), m), gm(d.get('monthly_ndre',{}), m), gm(d.get('monthly_vh',{}), m), gm(d.get('monthly_vv',{}), m)
        if not np.isnan(n_v): all_ndvis[m].append(n_v)
        if not np.isnan(nr_v): all_ndres[m].append(nr_v)
        if not np.isnan(vh_v): all_vhs[m].append(vh_v)
        if not np.isnan(vv_v): all_vvs[m].append(vv_v)

default_ndvi = [np.median(all_ndvis[m]) if all_ndvis[m] else 0.4 for m in range(1, 13)]
default_ndre = [np.median(all_ndres[m]) if all_ndres[m] else 0.3 for m in range(1, 13)]
default_vh   = [np.median(all_vhs[m]) if all_vhs[m] else -17.0 for m in range(1, 13)]
default_vv   = [np.median(all_vvs[m]) if all_vvs[m] else -11.0 for m in range(1, 13)]

# Build pure channel blocks
X_ndvi, X_ndre, X_sar, X_combined = [], [], [], []

for d in data:
    ndi = interp(np.array([gm(d.get('monthly_ndvi', {}), m) for m in range(1, 13)]), default_ndvi)
    nri = interp(np.array([gm(d.get('monthly_ndre', {}), m) for m in range(1, 13)]), default_ndre)
    vhi = interp(np.array([gm(d.get('monthly_vh', {}), m) for m in range(1, 13)]), default_vh)
    vvi = interp(np.array([gm(d.get('monthly_vv', {}), m) for m in range(1, 13)]), default_vv)
    
    X_ndvi.append(list(ndi))
    X_ndre.append(list(nri))
    X_sar.append(list(vhi) + list(vvi))
    X_combined.append(list(ndi) + list(nri) + list(vhi) + list(vvi))

le = LabelEncoder()
y = le.fit_transform(labels)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=== 7-Class Sensor Ablation Experiment ===")
experiment_sets = {
    "1. NDVI-Only Model (12 feats)": np.array(X_ndvi),
    "2. NDRE-Only Model (12 feats)": np.array(X_ndre),
    "3. SAR-Only Model  (24 feats)": np.array(X_sar),
    "4. Multimodal Fusion (48 feats)": np.array(X_combined)
}

for name, X_arr in experiment_sets.items():
    model = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.06, loss_function='MultiClass', verbose=0, random_seed=42)
    scores = cross_val_score(model, X_arr, y, cv=cv, scoring='accuracy')
    print(f"{name:<32} | Baseline Accuracy: {scores.mean()*100:.1f}% +/- {scores.std()*100:.1f}%")
