"""
app.py
------
FastAPI application serving three pages:
  /          -> Home: combined summary of both claim types
  /property  -> Property claims deep-dive
  /casualty  -> Casualty claims deep-dive

And the forecast API both pages call:
  /api/forecast?claim_type=property|casualty&...scenario params
  /api/home   -> combined default-scenario summary for the home page
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import model as m

app = FastAPI(title="Sedgwick Claims Cost Inflation Index")

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _read(name: str) -> str:
    return (BASE_DIR / "templates" / name).read_text()


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=_read("home.html"))


@app.get("/property", response_class=HTMLResponse)
async def property_page():
    return HTMLResponse(content=_read("property.html"))


@app.get("/casualty", response_class=HTMLResponse)
async def casualty_page():
    return HTMLResponse(content=_read("casualty.html"))


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/home")
async def api_home():
    try:
        return JSONResponse(content=m.get_home_summary())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/forecast")
async def api_forecast(
    claim_type:        str   = Query(default="casualty", regex="^(casualty|property)$"),
    horizon_months:    int   = Query(default=36, ge=6, le=60),
    cpi_pct:           float = Query(default=2.5, ge=0, le=15.0),
    severity_pct:      float = Query(default=5.0, ge=-10, le=30),
    legal_mult:        float = Query(default=1.2, ge=1.0, le=2.0),
    covid_adj:         float = Query(default=0.0, ge=-20, le=20),
    seasonal_adj:      float = Query(default=0.0, ge=-15, le=15),
    peril_group:       str   = Query(default="ALL"),
    severity_group:    str   = Query(default="ALL"),
    claims_subset:     str   = Query(default="settled", regex="^(settled|all)$"),
    building_contents: str   = Query(default="ALL", regex="^(ALL|Buildings|Contents)$"),
    claim_category:    str   = Query(default="ALL", regex="^(ALL|Domestic|Commercial)$"),
    region:            str   = Query(default="ALL"),
):
    try:
        # Convert months to years (ceiling) for the model, then trim output to exact months
        import math
        horizon_years = math.ceil(horizon_months / 12)
        result = m.get_forecast(
            claim_type=claim_type,
            horizon_years=horizon_years,
            cpi_pct=cpi_pct,
            severity_pct=severity_pct,
            legal_mult=legal_mult,
            covid_adj=covid_adj,
            seasonal_adj=seasonal_adj,
            peril_group=peril_group,
            severity_group=severity_group,
            claims_subset=claims_subset,
            building_contents=building_contents,
            claim_category=claim_category,
            region=region,
        )
        # Trim forecasts to exact months
        if "prophet_forecast" in result:  # settled claims
            result["prophet_forecast"] = result["prophet_forecast"][:horizon_months]
            result["sarimax_forecast"] = result["sarimax_forecast"][:horizon_months]
        else:  # all claims (has separate cost/count)
            result["prophet_forecast_cost"] = result["prophet_forecast_cost"][:horizon_months]
            result["sarimax_forecast_cost"] = result["sarimax_forecast_cost"][:horizon_months]
            result["prophet_forecast_count"] = result["prophet_forecast_count"][:horizon_months]
            result["sarimax_forecast_count"] = result["sarimax_forecast_count"][:horizon_months]
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/retrain")
async def retrain():
    cache = Path(__file__).parent / "model_cache.pkl"
    cache.unlink(missing_ok=True)
    result = m.get_forecast("casualty")
    result2 = m.get_forecast("property")
    return {"status": "retrained", "casualty_kpis": result["kpis"], "property_kpis": result2["kpis"]}
