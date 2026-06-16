import json,numpy as np,warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold,cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix
from collections import Counter

with open('/workspaces/crop-trajectory/data/psetae_100_fullseason.json') as f:
    data=json.load(f)

CM={'Grassland':'Grass','Barley':'Barley','Wheat':'Wheat',
    'Oats':'Other','Oilseed Rape':'Other','Maize':'Other',
    'Beans':'Other','Potatoes':'Other'}
data=[d for d in data if d['label'] in CM]
labels=[CM[d['label']] for d in data]
print('Classes:',Counter(labels))

def gm(dic,m):
    v=dic.get(str(m),dic.get(m,0))
    return float(v) if v else np.nan

def featurize(d):
    nd=np.array([gm(d['monthly_ndvi'],m) for m in range(1,13)])
    vh=np.array([gm(d['monthly_vh'],m) for m in range(1,13)])
    nr=np.array([gm(d['monthly_ndre'],m) for m in range(1,13)])
    vl=np.where(~np.isnan(vh),10**(np.clip(vh,-50,0)/10),np.nan)
    vn=nd[~np.isnan(nd)]; vv=vl[~np.isnan(vl)]
    def s(v):
        try: f=float(v); return np.nan if not np.isfinite(f) else f
        except: return np.nan
    pm=float(np.nanargmax(np.nan_to_num(nd,nan=-99))+1)
    rn=s(np.max(vn)-np.min(vn)) if len(vn)>1 else np.nan
    mv=s(np.mean(vv)) if len(vv)>0 else np.nan
    sv=s(np.std(vv))  if len(vv)>1 else np.nan
    pk=s(np.max(vn))  if len(vn)>0 else np.nan
    mn=s(np.min(vn))  if len(vn)>0 else np.nan
    gu=s(nd[4]-nd[0]) if not(np.isnan(nd[4]) or np.isnan(nd[0])) else np.nan
    hd=s(nd[4]-nd[7]) if not(np.isnan(nd[4]) or np.isnan(nd[7])) else np.nan
    rc=s(nd[8]-nd[7]) if not(np.isnan(nd[8]) or np.isnan(nd[7])) else np.nan
    os2=s(nd[3]-nd[5]) if not(np.isnan(nd[3]) or np.isnan(nd[5])) else np.nan
    nm=s(nr[4]); vm=s(vl[4]); va=s(vl[7])
    sd=s(vl[4]-vl[7]) if not(np.isnan(vl[4]) or np.isnan(vl[7])) else np.nan
    ar=float(d.get('area_ha',0))
    return [pm,rn,mv,sv,pk,mn,s(nd[0]),s(nd[2]),s(nd[3]),s(nd[4]),
            s(nd[5]),s(nd[6]),s(nd[7]),s(nd[8]),s(nd[9]),
            gu,hd,rc,os2,nm,vm,va,sd,ar]

X=np.array([featurize(d) for d in data],dtype=np.float64)
X=np.where(np.isinf(X),np.nan,X)
print(f'Shape:{X.shape} NaN:{np.isnan(X).sum()}')
le=LabelEncoder(); y=le.fit_transform(labels)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
from imblearn.ensemble import BalancedRandomForestClassifier
mdl=Pipeline([('imp',SimpleImputer(strategy='median')),
              ('clf',BalancedRandomForestClassifier(
               n_estimators=500,max_depth=10,random_state=42,n_jobs=-1))])
sc=cross_val_score(mdl,X,y,cv=cv,scoring='accuracy')
yp=cross_val_predict(mdl,X,y,cv=cv)
cm=confusion_matrix(y,yp)
print(f'BRF:{round(sc.mean()*100)}% folds:{[round(s*100) for s in sc]}')
for i,c in enumerate(le.classes_):
    a=round(cm[i,i]/max(cm[i].sum(),1)*100)
    wr=' '.join(f"{le.classes_[j][:4]}:{cm[i,j]}" for j in range(len(le.classes_)) if j!=i and cm[i,j]>0)
    print(f'  {c:<12} {a:>3}%  {wr}')
print('Previous best: 65%')