
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
    return float(v) if v else np.nan

def regional_medians(all_data, key):
    meds = []
    for m in range(1,13):
        vals = [float(d[key][k]) for d in all_data
                for k in [str(m), m] if k in d[key] and d[key][k]]
        meds.append(float(np.median(vals)) if vals else 0.0)
    return np.array(meds)

ndvi_reg = regional_medians(data, 'monthly_ndvi')
vh_reg   = regional_medians(data, 'monthly_vh')
ndre_reg = regional_medians(data, 'monthly_ndre')

def interp(arr, fallback):
    x = np.arange(12)
    mask = ~np.isnan(arr)
    if mask.sum() >= 2:
        return np.interp(x, x[mask], arr[mask])
    elif mask.sum() == 1:
        return np.full(12, arr[mask][0])
    return fallback.copy()

def featurize(d):
    nd = np.array([gm(d['monthly_ndvi'], m) for m in range(1,13)])
    vh = np.array([gm(d['monthly_vh'],   m) for m in range(1,13)])
    nr = np.array([gm(d['monthly_ndre'], m) for m in range(1,13)])

    vhl = np.where(~np.isnan(vh), 10**(np.clip(vh,-50,0)/10), np.nan)
    vvl_raw = np.array([gm(d['monthly_vv'], m) for m in range(1,13)])
    vvl = np.where(~np.isnan(vvl_raw), 10**(np.clip(vvl_raw,-50,0)/10), np.nan)

    # Interpolated arrays — no missing values
    ndi = interp(nd,  ndvi_reg)
    nri = interp(nr,  ndre_reg)
    vli = interp(vhl, 10**(np.clip(vh_reg,-50,0)/10))

    # SAR ratio (cloud-immune)
    mask = (vli > 1e-10)
    ratio = np.where(mask & ~np.isnan(vvl), vvl/(vli+1e-10), np.nan)
    ratio_filled = interp(ratio, np.full(12, 1.0))

    def s(v):
        try:
            f = float(v)
            return np.nan if not np.isfinite(f) else f
        except:
            return np.nan

    # === 6 CORE PHENOLOGICAL STATISTICS ===

    # 1. Peak NDVI month — separates early (OSR=Apr, WW=May) from late (SB=Jun, Maize=Sep)
    f1 = float(np.argmax(ndi) + 1)

    # 2. NDVI amplitude — grassland low (~0.3), arable high (~0.7)
    f2 = s(np.max(ndi) - np.min(ndi))

    # 3. Winter greenness (Jan NDVI) — WW=0.6, SB=0.3, Grassland=0.8
    f3 = s(ndi[0])

    # 4. Harvest drop (May→Aug) — cereals crash, grassland stable, maize rises
    f4 = s(ndi[4] - ndi[7])

    # 5. SAR VV/VH ratio at heading (May) — barley awns cause high VH scatter
    f5 = s(ratio_filled[4])

    # 6. Post-harvest SAR (Aug VH linear) — bare soil after harvest = low VH
    f6 = s(vli[7])

    # Additional discriminating features
    f7  = s(ndi[3])              # Apr NDVI — OSR yellow peak
    f8  = s(ndi[8])              # Sep NDVI — maize still green, cereals bare
    f9  = s(ndi[4] - ndi[0])    # green-up speed
    f10 = s(ndi[3] - ndi[5])    # OSR signal (Apr peak then drops Jun)
    f11 = s(np.argmin(ndi) + 1) # trough month
    f12 = s(np.max(ndi))        # peak NDVI value
    f13 = s(np.std(vli))        # SAR temporal variance
    f14 = s(vli[4])             # May VH — heading biomass
    f15 = s(vli[4] - vli[7])    # SAR harvest drop
    f16 = s(ratio_filled[7])    # Aug VV/VH — post-harvest structure
    f17 = s(nri[4])             # May NDRE — chlorophyll content
    f18 = s(nri[3])             # Apr NDRE — OSR high NDRE
    f19 = s(ndi[10])            # Nov NDVI — winter crop emergence
    f20 = s(ndi[9] - ndi[7])    # Oct recovery
    f21 = s(np.mean(vli))       # mean SAR backscatter
    f22 = float(d.get('area_ha', 0))

    return [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,
            f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22]

X = np.array([featurize(d) for d in data], dtype=np.float64)
X = np.where(np.isinf(X), np.nan, X)
print(f'Shape:{X.shape} NaN:{np.isnan(X).sum()}')

le = LabelEncoder()
y  = le.fit_transform(labels)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, mdl in [
    ('BRF', Pipeline([('imp',SimpleImputer(strategy='median')),
                      ('clf',BalancedRandomForestClassifier(
                          n_estimators=500,max_depth=12,random_state=42,n_jobs=-1))])),
    ('XGB', Pipeline([('imp',SimpleImputer(strategy='median')),
                      ('clf',XGBClassifier(n_estimators=300,max_depth=5,
                          learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
                          min_child_weight=2,eval_metric='mlogloss',
                          random_state=42,n_jobs=-1,verbosity=0))])),
]:
    sc = cross_val_score(mdl, X, y, cv=cv, scoring='accuracy')
    yp = cross_val_predict(mdl, X, y, cv=cv)
    cm = confusion_matrix(y, yp)
    print(f'\n{name}: {round(sc.mean()*100)}%  folds:{[round(s*100) for s in sc]}')
    for i,c in enumerate(le.classes_):
        a  = round(cm[i,i]/max(cm[i].sum(),1)*100)
        wr = ' '.join(f"{le.classes_[j][:4]}:{cm[i,j]}"
                      for j in range(len(le.classes_)) if j!=i and cm[i,j]>0)
        print(f'  {c:<15} {a:>3}%  {wr}')
print('\nPrevious 8-class: 57%')
