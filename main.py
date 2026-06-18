"""
Cube Earth Crop Intelligence API
Full CGM endpoint serving:
- Crop type detection
- Growth stage
- LAI + Biomass + Canopy
- Field health score
- Yield estimate + confidence
- Disease risk
- Soil moisture
- Thermal stress
- Farmer report in plain English
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
sys.path.insert(0, '/workspaces/crop-trajectory')

app = FastAPI(
    title="Cube Earth Crop Intelligence API",
    description="SAR + Optical + Thermal + Weather crop intelligence",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

CLIENT_ID = os.environ.get("CDSE_CLIENT_ID")
CLIENT_SECRET = os.environ.get("CDSE_CLIENT_SECRET")


class FieldRequest(BaseModel):
    lat: float
    lng: float
    crop: str = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Cube Earth Crop Intelligence",
        "version": "1.0.0",
        "sensors": [
            "Sentinel-1 SAR",
            "Sentinel-2 NDVI/NDRE",
            "ECOSTRESS/MODIS LST",
            "SMAP soil moisture",
            "Open-Meteo weather"
        ]
    }


@app.post("/cgm")
def crop_growth_model(request: FieldRequest):
    """
    Full Crop Growth Model for a field location.
    Returns crop type, growth stage, LAI, biomass,
    yield estimate, weather and management alerts.
    """
    try:
        from models.cgm import run_cgm
        result = run_cgm(
            request.lat,
            request.lng,
            CLIENT_ID,
            CLIENT_SECRET
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/field_report")
def field_report(request: FieldRequest):
    """
    Complete farmer-friendly field report.
    Plain English decisions and recommendations.
    """
    try:
        from models.cgm import run_cgm
        from models.field_intelligence import get_field_intelligence
        from models.farmer_report import generate_farmer_report

        cgm = run_cgm(
            request.lat,
            request.lng,
            CLIENT_ID,
            CLIENT_SECRET
        )
        intel = get_field_intelligence(cgm, request.lat, request.lng)
        report = generate_farmer_report(cgm, intel)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/crop_stage")
def crop_stage(request: FieldRequest):
    """
    Quick crop stage detection only.
    Faster than full CGM — SAR only.
    """
    try:
        from models.crop_classifier import full_field_analysis
        from extractors.sar_timeseries import get_sar_timeseries
        from datetime import datetime

        today = datetime.now()
        season_start = datetime(
            today.year-1, 10, 1).strftime("%Y-%m-%d")
        season_end = today.strftime("%Y-%m-%d")

        obs = get_sar_timeseries(
            request.lat, request.lng,
            season_start, season_end,
            CLIENT_ID, CLIENT_SECRET,
            interval_days=12
        )
        available = [o for o in obs if o.get("available")]
        result = full_field_analysis(available)
        return result["field_analysis"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/soil_moisture")
def soil_moisture(request: FieldRequest):
    """
    Root zone soil moisture profile.
    FAO-56 water balance + SAR correction.
    """
    try:
        from models.soil_moisture import get_soil_moisture_profile
        result = get_soil_moisture_profile(
            request.lat,
            request.lng,
            request.crop or "Unknown"
        )
        if result.get("error"):
            # Return partial data with weather-only estimate
            import requests as _req
            r = _req.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": request.lat,
                    "longitude": request.lng,
                    "daily": "precipitation_sum,et0_fao_evapotranspiration,soil_moisture_0_to_7cm_mean",
                    "past_days": 10,
                    "forecast_days": 1,
                    "timezone": "auto"
                }, timeout=45)
            if r.status_code == 200:
                daily = r.json().get("daily", {})
                sm = daily.get("soil_moisture_0_to_7cm_mean", [])
                rain = daily.get("precipitation_sum", [])
                valid_sm = [x for x in sm if x]
                current = valid_sm[-1] if valid_sm else None
                return {
                    "data_sources": {
                        "surface_moisture": "Open-Meteo soil model",
                        "root_zone": "Open-Meteo direct",
                        "sar_corrections_applied": False
                    },
                    "root_zone": {
                        "current_moisture": current,
                        "status": "Adequate" if current and current > 0.2 else "Unknown",
                        "status_icon": "✅" if current and current > 0.2 else "❓",
                        "trend": "Stable",
                        "trend_icon": "→",
                        "irrigation_needed": False
                    },
                    "irrigation_recommendation": "No irrigation needed" if current and current > 0.2 else "Data unavailable"
                }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/thermal_stress")
def thermal_stress(request: FieldRequest):
    """
    Thermal stress from ECOSTRESS/MODIS LST.
    CWSI and actual ET estimation.
    """
    try:
        from extractors.ecostress import get_thermal_stress_profile
        result = get_thermal_stress_profile(
            request.lat,
            request.lng,
            crop_type=request.crop or "Unknown"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PolygonRequest(BaseModel):
    polygon: list
    crop: str = None
    sowing_date: str = None
    start_date: str = None
    end_date: str = None
    interval_days: int = 12


@app.post("/sar_polygon")
def sar_polygon(request: PolygonRequest):
    """
    SAR time series for a field polygon.
    Returns field-level zonal statistics.
    Input: GeoJSON polygon coordinates [[lng,lat],...]
    """
    try:
        from extractors.sar_polygon import get_sar_timeseries_polygon
        from datetime import datetime

        today = datetime.now()
        start = request.start_date or             datetime(today.year-1, 10, 1).strftime("%Y-%m-%d")
        end = request.end_date or today.strftime("%Y-%m-%d")

        results = get_sar_timeseries_polygon(
            request.polygon,
            start, end,
            CLIENT_ID, CLIENT_SECRET,
            request.interval_days
        )
        available = [r for r in results if r.get("available")]
        return {
            "observations": len(available),
            "start_date": start,
            "end_date": end,
            "polygon_vertices": len(request.polygon),
            "data": available
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/field_analysis_polygon")
def field_analysis_polygon(request: PolygonRequest):
    """
    Full crop analysis for a field polygon.
    More accurate than point-based analysis.
    """
    try:
        from extractors.sar_polygon import get_sar_timeseries_polygon
        from models.crop_classifier import full_field_analysis
        from datetime import datetime

        today = datetime.now()
        start = request.start_date or             datetime(today.year-1, 10, 1).strftime("%Y-%m-%d")
        end = request.end_date or today.strftime("%Y-%m-%d")

        # Get polygon SAR data
        results = get_sar_timeseries_polygon(
            request.polygon,
            start, end,
            CLIENT_ID, CLIENT_SECRET,
            request.interval_days
        )
        available = [r for r in results if r.get("available")]

        if not available:
            raise HTTPException(
                status_code=404,
                detail="No SAR data available for this polygon")

        # Run crop analysis
        analysis = full_field_analysis(available)

        # Add field variability summary
        variability = [r.get("field_variability", 0)
                      for r in available]
        avg_var = round(sum(variability)/len(variability), 4)             if variability else None

        analysis["field_variability"] = {
            "average": avg_var,
            "interpretation": (
                "Uniform crop development"
                if avg_var and avg_var < 0.3 else
                "Moderate within-field variation"
                if avg_var and avg_var < 0.5 else
                "High within-field variation — check for patches"
            )
        }

        return analysis
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500, 
            detail=f"{str(e)} | {traceback.format_exc()[-500:]}")




@app.get("/ecostress_et")
async def ecostress_et(lat: float, lng: float, days: int = 90):
    """
    Query ECOSTRESS L3 ET via NASA AppEEARS
    Returns evapotranspiration time series for a location
    """
    import requests as req
    from datetime import datetime, timedelta

    username = os.environ.get("NASA_EARTHDATA_USER", "")
    password = os.environ.get("NASA_EARTHDATA_PASS", "")
    APPEEARS = "https://appeears.earthdatacloud.nasa.gov/api"

    if not username or not password:
        return {"error": "NASA_EARTHDATA_USER and NASA_EARTHDATA_PASS not set"}

    # Get fresh AppEEARS session token
    login_r = req.post(f"{APPEEARS}/login",
                       auth=(username, password), timeout=30)
    if login_r.status_code != 200:
        return {"error": f"AppEEARS login failed: {login_r.status_code}",
                "detail": login_r.text[:200]}
    token = login_r.json().get("token")
    if not token:
        return {"error": "No token returned from AppEEARS login"}

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Submit point sample request
    task = {
        "task_type": "point",
        "task_name": f"cubeearth_et_{end_date.strftime('%Y%m%d')}",
        "params": {
            "dates": [{
                "startDate": start_date.strftime("%m-%d-%Y"),
                "endDate": end_date.strftime("%m-%d-%Y")
            }],
            "layers": [
                {"product": "ECO_L3T_JET.002", "layer": "ETdaily"},
                {"product": "ECO_L3T_JET.002", "layer": "PTJPLSMinst"},
                {"product": "ECO_L3T_JET.002", "layer": "ETinstUncertainty"},
                {"product": "ECO_L3T_JET.002", "layer": "cloud"}
            ],
            "coordinates": [{
                "id": "parcel",
                "latitude": lat,
                "longitude": lng,
                "category": "field"
            }]
        }
    }

    headers = {"Authorization": f"Bearer {token}"}

    try:
        # Submit task
        r = req.post(f"{APPEEARS}/task", json=task,
                     headers=headers, timeout=30)
        if r.status_code not in [200, 202]:
            return {"error": f"Task submission failed: {r.status_code}",
                    "detail": r.text[:200]}

        task_id = r.json().get("task_id")

        # Return immediately — AppEEARS takes 5-15 min
        # Render times out at 30s so don't poll here
        return {
            "status": "submitted",
            "task_id": task_id,
            "lat": lat,
            "lng": lng,
            "period_days": days,
            "message": "Task submitted. Poll for results every 2 min.",
            "poll_url": f"/ecostress_task?id={task_id}",
            "results_url": f"/ecostress_results?id={task_id}"
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/ecostress_layers")
async def ecostress_layers(product: str = "ECO_L3T_JET.002"):
    """List layers for a specific ECOSTRESS product"""
    import requests as req
    username = os.environ.get("NASA_EARTHDATA_USER", "")
    password = os.environ.get("NASA_EARTHDATA_PASS", "")
    APPEEARS = "https://appeears.earthdatacloud.nasa.gov/api"
    login_r = req.post(f"{APPEEARS}/login", auth=(username, password), timeout=30)
    token = login_r.json().get("token")
    r = req.get(f"{APPEEARS}/product/{product}",
                headers={"Authorization": f"Bearer {token}"}, timeout=30)
    return {"product": product, "layers": r.json()}


@app.get("/ecostress_products")
async def ecostress_products():
    """List all available ECOSTRESS products"""
    import requests as req
    username = os.environ.get("NASA_EARTHDATA_USER", "")
    password = os.environ.get("NASA_EARTHDATA_PASS", "")
    APPEEARS = "https://appeears.earthdatacloud.nasa.gov/api"
    login_r = req.post(f"{APPEEARS}/login", auth=(username, password), timeout=30)
    token = login_r.json().get("token")
    r = req.get(f"{APPEEARS}/product",
                headers={"Authorization": f"Bearer {token}"}, timeout=30)
    products = r.json()
    eco = [{"product": p["ProductAndVersion"],
            "description": p.get("Description",""),
            "resolution": p.get("Resolution","")}
           for p in products if "ECO" in p.get("ProductAndVersion","")]
    return {"ecostress_products": eco, "total": len(eco)}


@app.get("/ecostress_results")
async def ecostress_results(id: str):
    """Download and parse results from a completed AppEEARS task"""
    import requests as req
    username = os.environ.get("NASA_EARTHDATA_USER", "")
    password = os.environ.get("NASA_EARTHDATA_PASS", "")
    APPEEARS = "https://appeears.earthdatacloud.nasa.gov/api"
    login_r = req.post(f"{APPEEARS}/login", auth=(username, password), timeout=30)
    token = login_r.json().get("token")
    headers = {"Authorization": f"Bearer {token}"}

    # Get file list
    bundle = req.get(f"{APPEEARS}/bundle/{id}", headers=headers, timeout=15).json()
    files = bundle.get("files", [])

    csv_file = next((f for f in files if f["file_name"].endswith("-results.csv")), None)
    if not csv_file:
        return {"error": "No results CSV found", "files": [f["file_name"] for f in files]}

    # Download CSV
    csv_r = req.get(
        f"{APPEEARS}/bundle/{id}/{csv_file['file_id']}",
        headers=headers, timeout=60
    )
    text = csv_r.text.strip()
    if not text:
        return {"error": "Empty CSV", "task_id": id,
                "note": "No ECOSTRESS data at this location/time — outside ISS orbit coverage"}

    lines = text.splitlines()
    if len(lines) < 2:
        return {"error": "No data rows", "header": lines[0] if lines else ""}

    header = lines[0].split(",")
    observations = []
    for line in lines[1:]:
        if not line.strip(): continue
        vals = line.split(",")
        if len(vals) != len(header): continue
        row = dict(zip(header, vals))
        def sf(k):
            try: v=float(row.get(k,"nan")); return None if v!=v else round(v,4)
            except: return None
        obs = {
            "date": row.get("Date",""),
            "tile": row.get("ECOSTRESS_Tile",""),
            "et_daily_wm2": sf("ECO_L3T_JET_002_ETdaily"),
            "et_inst_wm2":  sf("ECO_L3T_JET_002_PTJPLSMinst"),
            "et_uncertainty": sf("ECO_L3T_JET_002_ETinstUncertainty"),
            "cloud": sf("ECO_L3T_JET_002_cloud"),
            "lat": sf("Latitude"),
            "lng": sf("Longitude")
        }
        # Include all rows — even cloud covered (et will be None)
        observations.append(obs)

    et_vals = [o["et_daily_wm2"] for o in observations if o["et_daily_wm2"]]
    return {
        "task_id": id,
        "total_rows": len(lines)-1,
        "valid_et_observations": len(observations),
        "et_summary": {
            "mean_wm2": round(sum(et_vals)/len(et_vals),2) if et_vals else None,
            "min_wm2":  round(min(et_vals),2) if et_vals else None,
            "max_wm2":  round(max(et_vals),2) if et_vals else None,
        } if et_vals else None,
        "observations": observations[:20],
        "source": "ECOSTRESS ECO_L3T_JET.002 via NASA AppEEARS",
        "csv_preview": lines[:3]
    }


@app.get("/ecostress_task")
async def ecostress_task_status(id: str):
    """Check status of a pending AppEEARS task"""
    import requests as req
    username = os.environ.get("NASA_EARTHDATA_USER", "")
    password = os.environ.get("NASA_EARTHDATA_PASS", "")
    APPEEARS = "https://appeears.earthdatacloud.nasa.gov/api"
    login_r = req.post(f"{APPEEARS}/login", auth=(username, password), timeout=30)
    token = login_r.json().get("token")
    r = req.get(f"{APPEEARS}/task/{id}",
                headers={"Authorization": f"Bearer {token}"}, timeout=15)
    data = r.json()
    # If done, also fetch download links
    if data.get("status") == "done":
        bundle = req.get(f"{APPEEARS}/bundle/{id}",
                        headers={"Authorization": f"Bearer {token}"}, timeout=15)
        files = bundle.json().get("files", [])
        data["files"] = [{"name": f["file_name"],
                          "size_mb": round(f.get("file_size",0)/1024/1024, 2)}
                         for f in files]
    return data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)


class ParcelRequest(BaseModel):
    parcel_id: str = None
    lat: float
    lng: float
    bbox_km: float = 1.0
    crop_hint: str = None


@app.post("/parcel_intelligence")
def parcel_intelligence(request: ParcelRequest):
    """
    Full crop intelligence for an Irish DAFM parcel.
    Fetches real parcel polygon then runs full analysis.
    """
    try:
        import requests as _req
        from models.crop_classifier import full_field_analysis
        from extractors.sar_polygon import get_sar_timeseries_polygon
        from datetime import datetime

        # Step 1 — Get DAFM parcel polygon
        delta = request.bbox_km / 111.0
        bbox_url = (
            f"https://cube-earth.onrender.com/parcels_in_bbox"
            f"?minlng={request.lng - delta}"
            f"&minlat={request.lat - delta}"
            f"&maxlng={request.lng + delta}"
            f"&maxlat={request.lat + delta}"
        )

        r = _req.get(bbox_url, timeout=30)
        if r.status_code != 200:
            raise HTTPException(
                status_code=404,
                detail="Could not fetch DAFM parcels")

        features = r.json().get("features", [])
        if not features:
            raise HTTPException(
                status_code=404,
                detail="No DAFM parcels found at this location")

        # Find closest parcel to point or match parcel_id
        parcel = None
        if request.parcel_id:
            for f in features:
                props = f.get("properties", {})
                if request.parcel_id in [
                    props.get("PAR_LAB", ""),
                    props.get("HERD", "")
                ]:
                    parcel = f
                    break

        if not parcel:
            parcel = features[0]

        props = parcel["properties"]
        coords = parcel["geometry"]["coordinates"][0]
        crop = props.get("CROP", "Unknown")
        area = props.get("CLAIM_AREA", 0)
        # Frontend can pass known crop type as hint
        if request.crop_hint and (not crop or crop == "Unknown"):
            crop = request.crop_hint

        # Step 2 — SAR polygon extraction
        today = datetime.now()
        season_start = datetime(
            today.year-1, 10, 1).strftime("%Y-%m-%d")
        season_end = today.strftime("%Y-%m-%d")

        obs = get_sar_timeseries_polygon(
            coords, season_start, season_end,
            CLIENT_ID, CLIENT_SECRET,
            interval_days=12
        )
        available = [o for o in obs if o.get("available")]

        if not available:
            raise HTTPException(
                status_code=404,
                detail="No SAR data for this parcel")

        # Step 3 — Field variability (needed before crop analysis)
        variability = [o.get("field_variability", 0)
                      for o in available]
        avg_var = round(
            sum(variability)/len(variability), 4)             if variability else 0

        # Crop analysis
        # DAFM crop type mapping
        dafm_crop_map = {
            "Permanent Pasture": "Grassland",
            "Temporary Grassland": "Grassland",
            "Winter Wheat": "Winter Wheat",
            "Spring Barley": "Spring Barley",
            "Winter Barley": "Winter Wheat",
            "Oats": "Spring Barley",
            "Winter Oilseed Rape": "Oilseed Rape",
            "Potatoes": "Potato",
            "Woodland": "Grassland"
        }

        # Use Ireland data-driven classifier for unknown crops
        from models.crop_classifier_ireland import classify_ireland
        from models.crop_classifier import full_field_analysis
        from models.crop_identifier import analyse_time_series
        ireland_result = classify_ireland(available)
        sar_time_series = analyse_time_series(available)
        analysis = full_field_analysis(available)

        # CatBoost ML classifier — reuse pre-fetched SAR, no second network call
        try:
            from tools.inference_driver import predict_from_observations
            _par_id = props.get("PAR_LAB") or props.get("HERD") or "UNKNOWN"
            cat_result = predict_from_observations(
                parcel["geometry"]["coordinates"],
                CLIENT_ID, CLIENT_SECRET,
                parcel_id=_par_id,
                sar_observations=obs,
                area_ha=float(area or 5.0)
            )
            cat_crop_str = str(cat_result.get("crop_type", "Unknown"))
            cat_conf_pct = float(cat_result.get("confidence_pct", 0.0))
            cat_conf = cat_conf_pct / 100.0
            analysis["catboost_classifier"] = {
                "crop_type": cat_crop_str,
                "confidence_pct": cat_conf_pct,
                "tier": (
                    "Tier1" if cat_conf >= 0.60
                    else "Tier2" if cat_conf >= 0.45
                    else "Tier3"
                ),
                "model": "CatBoost 7-class, 540 parcels, Tier1=90% precision"
            }
        except Exception as e:
            analysis["catboost_classifier"] = {"error": str(e)}

        # Add SAR time series to response
        analysis["sar_time_series"] = {
            "crop_type": sar_time_series["crop_type"],
            "confidence": sar_time_series["confidence"],
            "growth_phase": sar_time_series["growth_phase"],
            "evidence": sar_time_series["evidence"],
            "key_events": sar_time_series["key_events"],
            "seasonal_vh": sar_time_series["seasonal_vh"],
            "vv_range": sar_time_series["vv_range"]
        }

        # Override with Ireland classifier if no DAFM crop match
        if crop not in dafm_crop_map:
            analysis["field_analysis"]["crop_type"] = ireland_result["crop_type"]
            analysis["field_analysis"]["classification_confidence"] = ireland_result["confidence_pct"]
            analysis["field_analysis"]["classification_reasons"] = ireland_result.get("classification_reasons", [])
            analysis["field_analysis"]["crop_source"] = ireland_result.get("signature_source", "Irish SAR classifier")

        # Compare DAFM declaration vs SAR time series
        sar_crop = sar_time_series["crop_type"]
        dafm_agrees_sar = (
            crop in dafm_crop_map and
            dafm_crop_map[crop] == sar_crop
        ) or (
            "Grass" in sar_crop and
            any(g in crop for g in ["Pasture", "Grass", "Woodland"])
        )
        analysis["data_agreement"] = {
            "dafm_2024": crop,
            "sar_current": sar_crop,
            "sar_confidence": sar_time_series["confidence"],
            "agrees": dafm_agrees_sar,
            "note": "DAFM and SAR agree" if dafm_agrees_sar else
                    f"DAFM says {crop} but SAR suggests {sar_crop} — farmer confirmation recommended"
        }

        if crop in dafm_crop_map:
            mapped_crop = dafm_crop_map[crop]
            analysis["field_analysis"]["crop_type"] = mapped_crop
            analysis["field_analysis"]["crop_source"] = "DAFM parcel data"
            analysis["field_analysis"]["classification_confidence"] = 100
            analysis["field_analysis"]["classification_reasons"] = [
                f"Crop type obtained from official DAFM parcel record: {crop}"
            ]
            # Rename confidence field for DAFM-sourced crops
            analysis["field_analysis"]["parcel_confidence"] = "HIGH"
            analysis["field_analysis"].pop("growth_model_confidence", None)

            # Run correct crop growth model based on DAFM crop type
            if mapped_crop == "Spring Barley":
                from models.spring_barley import get_spring_barley_stage
                barley_result = get_spring_barley_stage(available, weather_data if 'weather_data' in dir() else {})
                analysis["field_analysis"]["current_stage"] = barley_result.get("stage_label", "Spring Barley — Growing")
                analysis["field_analysis"]["yield_estimate_tha"] = barley_result.get("yield_estimate_tha")
                analysis["field_analysis"]["management_alerts"] = barley_result.get("management_alerts", [])
                analysis["growth_model"] = {"system": "spring_barley", "crop": "Spring Barley", **barley_result}

            elif mapped_crop == "Winter Wheat":
                from models.winter_wheat import get_winter_wheat_stage
                wheat_result = get_winter_wheat_stage(available, weather_data if 'weather_data' in dir() else {})
                analysis["field_analysis"]["current_stage"] = wheat_result.get("stage_label", "Winter Wheat — Growing")
                analysis["field_analysis"]["yield_estimate_tha"] = wheat_result.get("yield_estimate_tha")
                analysis["field_analysis"]["management_alerts"] = wheat_result.get("management_alerts", [])
                analysis["growth_model"] = {"system": "winter_wheat", "crop": "Winter Wheat", **wheat_result}

            elif mapped_crop == "Oilseed Rape":
                from models.oilseed_rape import get_osr_stage
                osr_result = get_osr_stage(available, weather_data if 'weather_data' in dir() else {})
                analysis["field_analysis"]["current_stage"] = osr_result.get("stage_label", "Oilseed Rape — Growing")
                analysis["field_analysis"]["yield_estimate_tha"] = osr_result.get("yield_estimate_tha")
                analysis["field_analysis"]["management_alerts"] = osr_result.get("management_alerts", [])
                analysis["growth_model"] = {"system": "oilseed_rape", "crop": "Oilseed Rape", **osr_result}

            elif mapped_crop == "Potato":
                from models.potato import get_potato_stage
                potato_result = get_potato_stage(available, weather_data if 'weather_data' in dir() else {})
                analysis["field_analysis"]["current_stage"] = potato_result.get("stage_label", "Potato — Growing")
                analysis["field_analysis"]["yield_estimate_tha"] = potato_result.get("yield_estimate_tha")
                analysis["field_analysis"]["management_alerts"] = potato_result.get("management_alerts", [])
                analysis["growth_model"] = {"system": "potato", "crop": "Potato", **potato_result}

            # Override growth model for grassland
            elif mapped_crop == "Grassland":
                # Calculate grass RVI trend
                rvi_vals = [o["rvi"] for o in available if o.get("rvi")]
                recent = rvi_vals[-3:] if len(rvi_vals) >= 3 else rvi_vals
                trend = "Growing" if len(recent) > 1 and recent[-1] > recent[0] else "Stable"
                avg_rvi = round(sum(rvi_vals)/len(rvi_vals), 4) if rvi_vals else None
                latest_rvi = rvi_vals[-1] if rvi_vals else None
                field_var = avg_var if avg_var else 0

                analysis["field_analysis"]["current_stage"] = f"Permanent Pasture — {trend}"
                analysis["field_analysis"]["yield_estimate_tha"] = None
                analysis["field_analysis"]["yield_range"] = None
                analysis["field_analysis"]["management_alerts"] = [
                    f"Grass RVI: {avg_rvi} — {'Good grazing cover' if avg_rvi and avg_rvi > 0.55 else 'Monitor grass growth'}",
                    "Variability indicates mixed sward condition" if field_var > 0.4 else "Uniform sward"
                ]
                # Replace arable growth model with clean grassland model
                grassland_model = {
                    "system": "grassland",
                    "crop": "Permanent Pasture",
                    "trend": trend,
                    "rvi_current": latest_rvi,
                    "rvi_seasonal_mean": avg_rvi,
                    "grazing_cover": (
                        "Good" if avg_rvi and avg_rvi > 0.55 else
                        "Moderate" if avg_rvi and avg_rvi > 0.45 else
                        "Poor"
                    ),
                    "field_variability": field_var,
                    "latest_observation": available[-1]["date"] if available else None
                }
                analysis["growth_model"] = grassland_model
                # Also update crop_intelligence growth_model key
                if "field_analysis" in analysis:
                    analysis["field_analysis"]["growth_model"] = grassland_model
        else:
            analysis["field_analysis"]["crop_source"] = "SAR classifier"

        # Step 4 — Field variability already calculated above

        return {
            "parcel": {
                "crop_dafm": crop,
                "area_ha": area,
                "vertices": len(coords),
                "observations": len(available)
            },
            "field_variability": {
                "average": avg_var,
                "interpretation": (
                    "Uniform crop development"
                    if avg_var is not None and avg_var < 0.35 else
                    "Moderate within-field variation"
                    if avg_var is not None and avg_var < 0.45 else
                    "High within-field variation"
                )
            },
            "crop_intelligence": analysis
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
