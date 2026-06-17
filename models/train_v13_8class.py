import json, numpy as np, warnings
warnings.filterwarnings('ignore')
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from collections import Counter

with open('/workspaces/crop-trajectory/data/dataset_535.json') as f:
    data = json.load(f)

labels = [d['label'] for d in data]
print('Classes:', Counter(labels))

def gm(dic, m):
    v = dic.get(str(m), dic.get(m, 0))
    return float(v) if v and float(v) != 0.0 else np.nan

def regional_medians(all_data, key):
    meds = []
    for m in range(1, 13):
        vals = [float(d[key][k]) for d in all_data for k in [str(m), m] if key in d and k in d[key] and d[key][k]]
        meds.append(float(np.median(vals)) if vals else 0.0)
    return np.array(meds)

ndvi_reg = regional_medians(data, 'monthly_ndvi')
vh_reg   = regional_medians(data, 'monthly_vh')
vv_reg   = regional_medians(data, 'monthly_vv')

def interp(arr, fallback):
    x = np.arange(12)
    mask = ~np.isnan(arr)
    if mask.sum() >= 2:
        return np.interp(x, x[mask], arr[mask])
    elif mask.sum() == 1:
        return np.full(12, arr[mask][0])
    return fallback.copy()

def featurize(d):
    nd = np.array([gm(d.get('monthly_ndvi', {}), m) for m in range(1, 13)])
    vh = np.array([gm(d.get('monthly_vh', {}), m) for m in range(1, 13)])
    vv = np.array([gm(d.get('monthly_vv', {}), m) for m in range(1, 13)])

    ndi = interp(nd, ndvi_reg)

    vh_clipped = np.where(~np.isnan(vh), np.clip(vh, -30, 0), np.nan)
    vv_clipped = np.where(~np.isnan(vv), np.clip(vv, -25, 0), np.nan)
    diff_db = np.where(~np.isnan(vh_clipped) & ~np.isnan(vv_clipped), vv_clipped - vh_clipped, np.nan)
    
    diff_reg = np.clip(vv_reg, -25, 0) - np.clip(vh_reg, -30, 0)
    diff_interp = interp(diff_db, diff_reg)

    return list(ndi) + list(diff_interp)

X = np.array([featurize(d) for d in data], dtype=np.float64)
print(f'True Matrix Shape: {X.shape} | Rogue NaNs: {np.isnan(X).sum()}')

le = LabelEncoder()
y  = le.fit_transform(labels)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in [
    ('BRF', BalancedRandomForestClassifier(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)),
    ('XGB', XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss', random_state=42, n_jobs=-1))
]:
    pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('clf', clf)])
    sc = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    yp = cross_val_predict(pipe, X, y, cv=cv)
    cm = confusion_matrix(y, yp)
    print(f'
{name}: {round(sc.mean()*100)}%  folds:{[round(s*100) for s in sc]}')
    for i, c in enumerate(le.classes_):
        acc = round(cm[i, i] / max(cm[i].sum(), 1) * 100)
        leaks = ' '.join(f"{le.classes_[j][:4]}:{cm[i, j]}" for j in range(len(le.classes_)) if j != i and cm[i, j] > 0)
        print(f'  {c:<15} {acc:>3}%  {leaks}')
