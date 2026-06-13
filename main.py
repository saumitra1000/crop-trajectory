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
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
