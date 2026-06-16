
import json, numpy as np, warnings
warnings.filterwarnings('ignore')
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from collections import Counter

with open('/workspaces/crop-trajectory/data/psetae_100_fullseason.json') as f:
    data = json.load(f)

CM = {'Grassland':'Grassland','Barley':'Barley','Wheat':'Wheat',
      'Oats':'Other','Oilseed Rape':'Other','Maize':'Other',
      'Beans':'Other','Potatoes':'Other'}
data   = [d for d in data if d['label'] in CM]
labels = [CM[d['label']] for d in data]
print('Classes:', Counter(labels))

def gm(dic, m):
    v = dic.get(str(m), dic.get(m, 0))
    return float(v) if v else np.nan

def interpolate(arr):
    x = np.arange(12)
    mask = ~np.isnan(arr)
    if mask.sum() >= 2:
        return np.interp(x, x[mask], arr[mask])
    return arr

def featurize(d):
    nd = np.array([gm(d['monthly_ndvi'], m) for m in range(1,13)])
    vh = np.array([gm(d['monthly_vh'],   m) for m in range(1,13)])
    vv = np.array([gm(d['monthly_vv'],   m) for m in range(1,13)])
    nr = np.array([gm(d['monthly_ndre'], m) for m in range(1,13)])

    # Interpolate NDVI to fill cloud gaps
    ndi = interpolate(nd)

    # SAR linear power
    vhl = np.where(~np.isnan(vh), 10**(np.clip(vh,-50,0)/10), np.nan)
    vvl = np.where(~np.isnan(vv), 10**(np.clip(vv,-50,0)/10), np.nan)
    mask = (~np.isnan(vhl)) & (~np.isnan(vvl)) & (vhl > 1e-10)
    ratio = np.where(mask, vvl/(vhl+1e-10), np.nan)

    vhl_v = vhl[~np.isnan(vhl)]
    ratio_v = ratio[~np.isnan(ratio)]

    def s(v):
        try:
            f = float(v)
            return np.nan if not np.isfinite(f) else f
        except:
            return np.nan

    # NDVI features using interpolated signal
    peak_m  = float(np.argmax(ndi) + 1)
    trough_m= float(np.argmin(ndi) + 1)
    pk      = s(np.max(ndi))
    mn      = s(np.min(ndi))
    rng     = s(pk - mn)
    gu      = s(ndi[4] - ndi[0])   # May - Jan green-up
    hd      = s(ndi[4] - ndi[7])   # May - Aug harvest drop
    rc      = s(ndi[9] - ndi[7])   # Oct - Aug recovery
    ww      = s(ndi[0])             # Jan (winter wheat stays green)
    osr     = s(ndi[3] - ndi[6])   # Apr - Jul (OSR early crash)
    maize   = s(ndi[8])             # Sep (maize late peak)
    n_jun   = s(nd[5])              # Raw Jun (not interpolated)
    n_jul   = s(nd[6])              # Raw Jul
    n_aug   = s(nd[7])              # Raw Aug

    # SAR structural features (cloud-free, always available)
    mean_vh = s(np.mean(vhl_v)) if len(vhl_v)>0 else np.nan
    std_vh  = s(np.std(vhl_v))  if len(vhl_v)>1 else np.nan
    # VV/VH ratio — key barley vs wheat discriminator
    mean_ratio = s(np.mean(ratio_v)) if len(ratio_v)>0 else np.nan
    # Ratio in May (heading stage — max structural difference)
    ratio_may  = s(ratio[4])
    ratio_aug  = s(ratio[7])
    # VH drop at harvest (cereal harvest = sudden VH decrease)
    vh_may = s(vhl[4]); vh_aug = s(vhl[7])
    vh_drop = s(vhl[4] - vhl[7]) if not (np.isnan(vhl[4]) or np.isnan(vhl[7])) else np.nan
    # SAR minimum month (harvest timing)
    vh_peak_m = float(np.nanargmax(np.nan_to_num(vhl, nan=-99)) + 1)
    # Temporal variance
    vh_range = s(np.max(vhl_v) - np.min(vhl_v)) if len(vhl_v)>1 else np.nan

    # NDRE (sensitive to nitrogen/chlorophyll — barley vs wheat differ)
    nr_apr = s(nr[3]); nr_may = s(nr[4])

    area = float(d.get('area_ha', 0))

    return [peak_m, trough_m, pk, mn, rng, gu, hd, rc, ww, osr, maize,
            n_jun, n_jul, n_aug,
            mean_vh, std_vh, mean_ratio, ratio_may, ratio_aug,
            vh_drop, vh_peak_m, vh_range, vh_may, vh_aug,
            nr_apr, nr_may, area]

X = np.array([featurize(d) for d in data], dtype=np.float64)
X = np.where(np.isinf(X), np.nan, X)
print(f'Shape:{X.shape} NaN:{np.isnan(X).sum()}')

le = LabelEncoder()
y  = le.fit_transform(labels)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

mdl = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('clf', BalancedRandomForestClassifier(
        n_estimators=500, max_depth=10,
        random_state=42, n_jobs=-1))
])

sc = cross_val_score(mdl, X, y, cv=cv, scoring='accuracy')
yp = cross_val_predict(mdl, X, y, cv=cv)
cm = confusion_matrix(y, yp)
print(f'BRF+SAR+interp: {round(sc.mean()*100)}%  folds:{[round(s*100) for s in sc]}')
for i,c in enumerate(le.classes_):
    a  = round(cm[i,i]/max(cm[i].sum(),1)*100)
    wr = ' '.join(f"{le.classes_[j][:4]}:{cm[i,j]}"
                  for j in range(len(le.classes_)) if j!=i and cm[i,j]>0)
    print(f'  {c:<12} {a:>3}%  {wr}')
print('Previous best: 65%')
