import json, os, sys, time
sys.path.insert(0, '/workspaces/crop-trajectory')
from extractors.fusion_extractor import extract_fusion_features
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

CLIENT_ID     = 'sh-6e5978f5-f5d6-43d6-874d-720d84121683'
CLIENT_SECRET = 'yrMEXQ5drlF26yrB4sTEXfWOIwKtB1fP'

CROP_MAP = {
    "Barley - Spring":"Barley","Barley - Winter":"Barley",
    "Wheat - Winter":"Wheat","Wheat - Spring":"Wheat",
    "Oats - Spring":"Oats","Oats - Winter":"Oats",
    "Oilseed Rape - Winter":"Oilseed Rape","Oilseed Rape - Spring":"Oilseed Rape",
    "Maize":"Maize","Beans - Spring":"Beans","Peas":"Beans",
    "Potatoes - Maincrop":"Potatoes","Rye":"Wheat",
    "Permanent Pasture":"Grassland",
}
TARGETS = {
    "Grassland":500,"Barley":500,"Wheat":200,"Oats":100,
    "Oilseed Rape":56,"Maize":85,"Beans":83,"Potatoes":39,
}

with open("data/dafm_arable_parcels.json") as f:
    parcels=json.load(f)

by_class=defaultdict(list)
for p in parcels:
    cls=CROP_MAP.get(p["properties"].get("CROP",""))
    if cls:
        p["properties"]["CROP_CLASS"]=cls
        by_class[cls].append(p)

per_class={}
for cls,target in TARGETS.items():
    take=min(len(by_class[cls]),target)
    per_class[cls]=by_class[cls][:take]
    print(f"  {cls:<15} {take}")
balanced=[]
max_len=max(len(v) for v in per_class.values())
for i in range(max_len):
    for cls in TARGETS:
        if i<len(per_class[cls]):
            balanced.append(per_class[cls][i])
print(f"  {'TOTAL':<15} {len(balanced)}")

CHECKPOINT="data/extraction_new_checkpoint.json"
results=[]
done_ids=set()
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        results=json.load(f)
    done_ids={r["par_lab"] for r in results}
    print(f"Resuming: {len(done_ids)} done")

failed=0
start=time.time()

for i,p in enumerate(balanced):
    par_lab=p["properties"].get("PAR_LAB","")
    if par_lab in done_ids: continue
    crop_class=p["properties"].get("CROP_CLASS","")
    polygon=p["geometry"]["coordinates"][0]
    lat=sum(c[1] for c in polygon)/len(polygon)
    lng=sum(c[0] for c in polygon)/len(polygon)

    try:
        feat=extract_fusion_features(polygon,CLIENT_ID,CLIENT_SECRET,
                                     start_date="2024-10-01",end_date="2025-09-30")
        if feat.get("n_sar",0)>=5:
            results.append({
                "par_lab":par_lab,"label":crop_class,
                "crop_raw":p["properties"].get("CROP",""),
                "lat":round(lat,5),"lng":round(lng,5),
                "area_ha":p["properties"].get("CLAIM_AREA",0),
                "features":feat["features"],
                "monthly_ndvi":feat["monthly_ndvi"],
                "monthly_ndre":feat["monthly_ndre"],
                "monthly_vh":feat["monthly_vh"],
                "monthly_vv":feat["monthly_vv"],
                "n_sar":feat["n_sar"],"n_ndvi":feat["n_ndvi"],
            })
            done_ids.add(par_lab)
        else:
            failed+=1
    except Exception as e:
        failed+=1
        print(f"  [{i}] error: {e}", flush=True)

    if (len(results)+failed)%10==0:
        with open(CHECKPOINT,"w") as f:
            json.dump(results,f)

    if i%5==0:
        elapsed=time.time()-start
        eta=(len(balanced)-i)/max((i+1)/elapsed,0.01)/60
        c=Counter(r["label"] for r in results)
        print(f"[{i:>4}/{len(balanced)}] done={len(results)} failed={failed} "
              f"eta={eta:.0f}min | {dict(c.most_common(3))}", flush=True)

with open(CHECKPOINT,"w") as f:
    json.dump(results,f)
with open("data/dataset_v2.json","w") as f:
    json.dump(results,f)
print(f"DONE: {len(results)} extracted, {failed} failed")
c=Counter(r["label"] for r in results)
for cls,count in c.most_common():
    print(f"  {cls:<15} {count}")
