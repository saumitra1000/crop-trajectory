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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)


class ParcelRequest(BaseModel):
    parcel_id: str = None
    lat: float
    lng: float
    bbox_km: float = 1.0


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
        # Use Ireland-specific classifier when DAFM parcel data available
        from models.crop_classifier_ireland import classify_ireland
        from models.crop_classifier import full_field_analysis
        ireland_result = classify_ireland(available)
        analysis = full_field_analysis(available)
        # Override with Ireland classifier if no DAFM crop match
        if crop not in dafm_crop_map:
            analysis["field_analysis"]["crop_type"] = ireland_result["crop_type"]
            analysis["field_analysis"]["classification_confidence"] = ireland_result["confidence_pct"]
            analysis["field_analysis"]["classification_reasons"] = ireland_result.get("classification_reasons", [])
            analysis["field_analysis"]["crop_source"] = ireland_result.get("signature_source", "Irish SAR classifier")
        
        # Override classifier with DAFM known crop
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

            # Override growth model for grassland
            if mapped_crop == "Grassland":
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
                analysis["growth_model"] = {
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
                    if avg_var and avg_var < 0.35 else
                    "Moderate within-field variation"
                    if avg_var and avg_var < 0.45 else
                    "High within-field variation"
                )
            },
            "crop_intelligence": analysis
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
