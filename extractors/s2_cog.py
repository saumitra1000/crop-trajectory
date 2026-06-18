"""
S2 COG Extractor — correct implementation
Uses:
  1. SCL mask to filter cloud/nodata/water
  2. Overview IFD (40m) for fast tile reads  
  3. 10m bands only (Red=B04, NIR=B08, RE=B05)
  4. Polygon vertex sampling
"""
import requests, struct, zlib, math, statistics
import numpy as np

NODATA_DN = 0
VALID_SCL = {4, 5, 7}  # vegetation, bare soil, unclassified

def latlon_to_utm(lat, lng):
    a=6378137.0;f=1/298.257223563;b=a*(1-f);e2=1-(b/a)**2
    k0=0.9996;zone=int((lng+180)/6)+1;lon0=math.radians((zone-1)*6-180+3)
    lat_r=math.radians(lat);lng_r=math.radians(lng)
    N=a/math.sqrt(1-e2*math.sin(lat_r)**2);T=math.tan(lat_r)**2
    A_=math.cos(lat_r)*(lng_r-lon0)
    M=a*((1-e2/4-3*e2**2/64)*lat_r-(3*e2/8+3*e2**2/32)*math.sin(2*lat_r))
    x=k0*N*(A_+(1-T)*A_**3/6)+500000;y=k0*(M+N*math.tan(lat_r)*(A_**2/2))
    return x,y

def fetch(url, s, e, timeout=10):
    r=requests.get(url, headers={"Range":f"bytes={s}-{e-1}"}, timeout=timeout)
    if r.status_code not in [200,206]:
        raise Exception(f"HTTP {r.status_code}")
    return r.content

def parse_ifd(hdr, ifd_offset, fmt="<"):
    n_tags=struct.unpack_from(f"{fmt}H",hdr,ifd_offset)[0]
    tags={}
    for i in range(n_tags):
        off=ifd_offset+2+i*12
        if off+12>len(hdr): break
        tag=struct.unpack_from(f"{fmt}H",hdr,off)[0]
        typ=struct.unpack_from(f"{fmt}H",hdr,off+2)[0]
        cnt=struct.unpack_from(f"{fmt}I",hdr,off+4)[0]
        val=struct.unpack_from(f"{fmt}I",hdr,off+8)[0]
        tags[tag]=(typ,cnt,val)
    # Next IFD offset
    next_off=ifd_offset+2+n_tags*12
    next_ifd=struct.unpack_from(f"{fmt}I",hdr,next_off)[0] if next_off+4<=len(hdr) else 0
    return tags, next_ifd

def get_ifd_meta(url, ifd_level=2):
    """Get metadata from specified IFD level (0=full res, 2=overview ~40m)"""
    fmt="<"
    hdr=fetch(url, 0, 65536)
    ifd_offset=struct.unpack_from(f"{fmt}I",hdr,4)[0]
    
    for level in range(ifd_level+1):
        tags, next_ifd = parse_ifd(hdr, ifd_offset, fmt)
        if level < ifd_level:
            if next_ifd == 0: break
            ifd_offset = next_ifd

    W=tags[256][2]; H=tags[257][2]
    tw=tags[322][2]; th=tags[323][2]
    top=tags[324][2]; nt2=tags[324][1]
    tbp=tags[325][2]
    bpe=8 if tags[324][0]==16 else 4
    fc="Q" if tags[324][0]==16 else "I"

    sd=fetch(url, tags[33550][2], tags[33550][2]+23)
    sx=struct.unpack_from(f"{fmt}d",sd,0)[0]
    td=fetch(url, tags[33922][2], tags[33922][2]+47)
    tx=struct.unpack_from(f"{fmt}d",td,24)[0]
    ty=struct.unpack_from(f"{fmt}d",td,32)[0]
    od=fetch(url, top, top+nt2*bpe)
    sd2=fetch(url, tbp, tbp+nt2*bpe)

    return dict(W=W,H=H,tw=tw,th=th,ta=(W+tw-1)//tw,
                nt2=nt2,od=od,sd2=sd2,sx=sx,sy=abs(sx),
                tx=tx,ty=ty,bpe=bpe,fc=fc,fmt=fmt,url=url)

_tile_cache = {}

def read_dn(meta, lat, lng):
    """Read DN at lat/lng using cached tile"""
    fmt=meta["fmt"]; tw=meta["tw"]; th=meta["th"]
    ux,uy=latlon_to_utm(lat,lng)
    col=int((ux-meta["tx"])/meta["sx"])
    row=int((meta["ty"]-uy)/meta["sy"])
    if not(0<=col<meta["W"] and 0<=row<meta["H"]): return None

    ti=(row//th)*meta["ta"]+(col//tw)
    cache_key=(meta["url"],ti)

    if cache_key not in _tile_cache:
        bpe=meta["bpe"]; fc=meta["fc"]
        to=struct.unpack_from(f"{fmt}{fc}",meta["od"],ti*bpe)[0]
        ts=struct.unpack_from(f"{fmt}{fc}",meta["sd2"],ti*bpe)[0]
        if ts==0:
            _tile_cache[cache_key]=None
        else:
            raw_t=fetch(meta["url"],to,to+ts)
            try: raw=zlib.decompress(raw_t)
            except:
                try: raw=zlib.decompress(raw_t,-15)
                except: raw=raw_t
            _tile_cache[cache_key]=raw

    raw=_tile_cache.get(cache_key)
    if raw is None: return None
    lc=col%tw; lr=row%th
    px=(lr*tw+lc)*2
    if px+2>len(raw): return None
    dn=struct.unpack_from(f"{fmt}H",raw,px)[0]
    return dn

def get_ndvi_for_parcel(polygon, scene, ifd_level=2):
    """
    Extract NDVI for a parcel from one S2 scene.
    Uses SCL mask + overview tiles + polygon vertex sampling.
    Returns median NDVI or None.
    """
    global _tile_cache
    _tile_cache = {}  # reset per scene

    assets = scene.get("assets",{})
    red_url = assets.get("red",{}).get("href","")
    nir_url = assets.get("nir",{}).get("href","")
    scl_url = assets.get("scl",{}).get("href","")
    re1_url = assets.get("rededge1",{}).get("href","")

    if not red_url or not nir_url: return None,None

    try:
        # Load metadata once per band (uses overview IFD)
        red_meta = get_ifd_meta(red_url, ifd_level)
        nir_meta = get_ifd_meta(nir_url, ifd_level)
        scl_meta = get_ifd_meta(scl_url, ifd_level) if scl_url else None
        re1_meta = get_ifd_meta(re1_url, ifd_level) if re1_url else None
    except Exception as e:
        return None, None

    # Sample points: evenly spaced polygon vertices + centroid
    lats=[c[1] for c in polygon]; lngs=[c[0] for c in polygon]
    lat_c=(min(lats)+max(lats))/2; lng_c=(min(lngs)+max(lngs))/2
    step=max(1,len(polygon)//8)
    pts=[(polygon[i][1],polygon[i][0]) for i in range(0,len(polygon),step)]
    pts.append((lat_c,lng_c))
    pts=pts[:10]  # max 10 points

    ndvi_vals=[]; ndre_vals=[]

    for lat,lng in pts:
        # Check SCL first
        if scl_meta:
            scl=read_dn(scl_meta,lat,lng)
            if scl is not None and scl not in VALID_SCL:
                continue  # cloud/water/shadow

        red=read_dn(red_meta,lat,lng)
        nir=read_dn(nir_meta,lat,lng)
        if red is None or nir is None: continue
        if red==NODATA_DN or nir==NODATA_DN: continue
        if red>60000 or nir>60000: continue  # nodata sentinel

        ndvi=(nir-red)/(nir+red+0.0001)
        if -0.5 < ndvi < 1.0:
            ndvi_vals.append(ndvi)

        if re1_meta:
            re1=read_dn(re1_meta,lat,lng)
            if re1 and re1!=NODATA_DN and re1<60000:
                ndre=(nir-re1)/(nir+re1+0.0001)
                if -0.5 < ndre < 1.0:
                    ndre_vals.append(ndre)

    ndvi_med = round(statistics.median(ndvi_vals),4) if ndvi_vals else None
    ndre_med = round(statistics.median(ndre_vals),4) if ndre_vals else None
    return ndvi_med, ndre_med


def get_optical_monthly_cog(polygon, start_date, end_date):
    """
    Extract monthly NDVI/NDRE for a parcel using S2 COG overviews.
    """
    lngs=[c[0] for c in polygon]; lats=[c[1] for c in polygon]
    bbox=[min(lngs),min(lats),max(lngs),max(lats)]
    lat_c=(bbox[1]+bbox[3])/2; lng_c=(bbox[0]+bbox[2])/2

    # Search S2 scenes
    r=requests.post("https://earth-search.aws.element84.com/v1/search",
        json={"collections":["sentinel-2-l2a"],"bbox":bbox,
              "datetime":f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
              "limit":200},timeout=15)
    if r.status_code!=200: return {},{}
    features=r.json().get("features",[])

    # Best scene per month
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

    monthly_ndvi={}; monthly_ndre={}
    for month,(cloud,scene) in monthly_candidates.items():
        ndvi,ndre=get_ndvi_for_parcel(polygon,scene,ifd_level=2)
        if ndvi is not None:
            monthly_ndvi[month]=ndvi
        if ndre is not None:
            monthly_ndre[month]=ndre

    return monthly_ndvi, monthly_ndre
