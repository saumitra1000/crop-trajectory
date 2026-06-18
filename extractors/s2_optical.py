import rasterio, json, requests, numpy as np, statistics
from rasterio.windows import Window
import pyproj
from concurrent.futures import ThreadPoolExecutor

VALID_SCL = {4, 5, 7}

def sample_band_bbox(url, pts_latlon, scl_url=None):
    vsi = f"/vsicurl/{url}"
    try:
        with rasterio.open(vsi) as src:
            transformer = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xy = [transformer.transform(lng, lat) for lat,lng in pts_latlon]
            rows_cols = []
            for x,y in xy:
                try:
                    r,c = src.index(x,y)
                    if 0<=r<src.height and 0<=c<src.width:
                        rows_cols.append((r,c))
                except: pass
            if not rows_cols: return []

            min_row = max(0, min(r for r,c in rows_cols)-2)
            max_row = min(src.height, max(r for r,c in rows_cols)+3)
            min_col = max(0, min(c for r,c in rows_cols)-2)
            max_col = min(src.width,  max(c for r,c in rows_cols)+3)
            win = Window(min_col, min_row, max_col-min_col, max_row-min_row)
            data = src.read(1, window=win)

        scl_mask = None
        if scl_url:
            try:
                with rasterio.open(f"/vsicurl/{scl_url}") as ss:
                    st = pyproj.Transformer.from_crs("EPSG:4326", ss.crs, always_xy=True)
                    sxy = [st.transform(lng, lat) for lat,lng in pts_latlon]
                    src2 = rasterio.open(vsi)
                    sr_cols = []
                    for x,y in sxy:
                        try:
                            r,c = ss.index(x,y)
                            if 0<=r<ss.height and 0<=c<ss.width:
                                sr_cols.append((r,c))
                            else:
                                sr_cols.append(None)
                        except: sr_cols.append(None)
                    if sr_cols:
                        valid_sr = [(r,c) for rc in sr_cols if rc for r,c in [rc]]
                        if valid_sr:
                            smr = max(0, min(r for r,c in valid_sr)-1)
                            smc = max(0, min(c for r,c in valid_sr)-1)
                            smxr = min(ss.height, max(r for r,c in valid_sr)+2)
                            smxc = min(ss.width,  max(c for r,c in valid_sr)+2)
                            sw = Window(smc, smr, smxc-smc, smxr-smr)
                            scl_data = ss.read(1, window=sw)
                            scl_mask = (sr_cols, scl_data, smr, smc)
            except: pass

        dns = []
        for i,(r,c) in enumerate(rows_cols):
            if scl_mask:
                sr_cols_list, scl_data, smr, smc = scl_mask
                if i < len(sr_cols_list) and sr_cols_list[i]:
                    sr,sc = sr_cols_list[i]
                    try:
                        scl_val = scl_data[sr-smr, sc-smc]
                        if scl_val not in VALID_SCL:
                            continue
                    except: pass

            lr = r - min_row
            lc = c - min_col
            patch = data[max(0,lr-1):lr+2, max(0,lc-1):lc+2].flatten()
            valid = patch[(patch > 0) & (patch < 60000)]
            if len(valid) > 0:
                dns.append(float(np.median(valid)))
        return dns
    except Exception:
        return []

def get_ndvi_scene(scene, pts):
    red_url = scene["assets"]["red"]["href"]
    nir_url = scene["assets"]["nir"]["href"]
    re1_url = scene["assets"].get("rededge1",{}).get("href","")
    scl_url = scene["assets"].get("scl",{}).get("href","")
    with ThreadPoolExecutor(max_workers=3) as ex:
        fr = ex.submit(sample_band_bbox, red_url, pts, scl_url)
        fn = ex.submit(sample_band_bbox, nir_url, pts, scl_url)
        fe = ex.submit(sample_band_bbox, re1_url, pts, scl_url) if re1_url else None
        red = fr.result(); nir = fn.result()
        re1 = fe.result() if fe else []
    ndvi = ndre = None
    if red and nir:
        vals = [(n-r)/(n+r+1e-4) for r,n in zip(red,nir) if n>0 and r>0]
        vals = [v for v in vals if -0.5<v<1.0]
        if vals: ndvi = round(statistics.median(vals),4)
    if re1 and nir:
        vals = [(n-r)/(n+r+1e-4) for r,n in zip(re1,nir) if n>0 and r>0]
        vals = [v for v in vals if -0.5<v<1.0]
        if vals: ndre = round(statistics.median(vals),4)
    return ndvi, ndre

def get_optical_monthly(polygon, start_date, end_date):
    lngs=[c[0] for c in polygon]; lats=[c[1] for c in polygon]
    bbox=[min(lngs),min(lats),max(lngs),max(lats)]
    lat_c=(bbox[1]+bbox[3])/2; lng_c=(bbox[0]+bbox[2])/2
    r=requests.post("https://earth-search.aws.element84.com/v1/search",
        json={"collections":["sentinel-2-l2a"],"bbox":bbox,
              "datetime":f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
              "limit":200},timeout=15)
    if r.status_code!=200: return {},{}
    features=r.json().get("features",[])
    monthly_candidates={}
    for f in features:
        date=f["properties"].get("datetime","")[:10]
        if not date: continue
        month=int(date.split("-")[1])
        cloud=f["properties"].get("eo:cloud_cover",100)
        if cloud>60: continue
        sb=f.get("bbox",[])
        if sb:
            margin=min(lat_c-sb[1],sb[3]-lat_c,lng_c-sb[0],sb[2]-lng_c)
            if margin<0.05: continue
        if month not in monthly_candidates or cloud<monthly_candidates[month][0]:
            monthly_candidates[month]=(cloud,f)
    # Use centroid + 4 interior offsets — fast, avoids large bbox
    lats=[c[1] for c in polygon]; lngs=[c[0] for c in polygon]
    lat_r=(max(lats)-min(lats))*0.2; lng_r=(max(lngs)-min(lngs))*0.2
    pts=[
        (lat_c,        lng_c),
        (lat_c+lat_r,  lng_c+lng_r),
        (lat_c-lat_r,  lng_c-lng_r),
        (lat_c+lat_r,  lng_c-lng_r),
        (lat_c-lat_r,  lng_c+lng_r),
    ]
    monthly_ndvi={}; monthly_ndre={}
    for month,(cloud,scene) in sorted(monthly_candidates.items()):
        try:
            ndvi,ndre=get_ndvi_scene(scene,pts)
            if ndvi is not None: monthly_ndvi[month]=ndvi
            if ndre is not None: monthly_ndre[month]=ndre
        except: pass
    return monthly_ndvi, monthly_ndre
