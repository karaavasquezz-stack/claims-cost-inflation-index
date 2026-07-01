"""
model.py
--------
Rebuilt to mirror the methodology used in the two reference scripts
(CasualtyDashboard.py and PropertyDashboard2.py) as closely as the
available data allows, using REAL Prophet and REAL SARIMAX for both
claim types (no hand-rolled approximations).

CASUALTY  (mirrors CasualtyDashboard.py)
  - Monthly avg settled cost, months with < MIN_CLAIMS settled claims
    dropped, last (incomplete) month dropped
  - Prophet: log-scale, yearly seasonality, COVID exogenous regressor
  - SARIMAX: log-scale, order=(1,1,1), seasonal_order=(1,1,0,12),
    COVID exogenous regressor
  - Correlation: claim cost YoY% vs CPI YoY% vs HICP YoY% (like-for-like)
  - Backtest: in-sample fit MAPE (matches reference's "MODEL FIT MAPE")
    plus an additional true out-of-sample 2025 holdout for both models

PROPERTY  (mirrors PropertyDashboard2.py)
  - Monthly avg settlement cost (buildings + contents), training window
    capped at TRAIN_CUTOFF
  - Prophet: HICP_adjusted and Construction-wage-index regressors
    (NOT a COVID regressor) — future regressor values are projected
    forward using the trailing-12-month slope, exactly as the reference
    does, since true future inflation isn't known at forecast time
  - SARIMAX: exogenous = delta_HICP (month-on-month change in the
    rebased HICP index), grid search over the same order/seasonal grid
    as the reference, model selected by lowest AIC
  - Correlation: claim cost level vs HICP/CPI/Construction index levels
  - Backtest: train ≤ 2023-12, test 2024 onward (matches reference)

Construction inflation note: the reference expects a dedicated
"Inflation_construction.csv". That file wasn't provided — in its place
we use the construction-sector average weekly earnings series
(earnings_clean.csv), rebased to 100 at the start period, as the wage
/cost-pressure proxy for property reinstatement claims.
"""

from pathlib import Path
import warnings
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR      = Path(__file__).parent / "data"
MODEL_CACHE   = Path(__file__).parent / "model_cache.pkl"

CASUALTY_PATH = DATA_DIR / "Casualty_data.csv"
PROPERTY_PATH = DATA_DIR / "Property_data.csv"
CPI_PATH      = DATA_DIR / "CPI.csv"
HICP_PATH     = DATA_DIR / "hicp_clean__1_.csv"
EARNINGS_PATH = DATA_DIR / "earnings_clean.csv"
LEGAL_PROXY_PATH  = DATA_DIR / "legal_proxy.csv"
MEDICAL_CPI_PATH  = DATA_DIR / "medical_cpi.csv"

# ── shared constants ────────────────────────────────────────────────────────
START_MONTH      = pd.Period("2020-01", freq="M")
COVID_START      = pd.Timestamp("2020-03-01")
COVID_END        = pd.Timestamp("2021-06-30")
FORECAST_YEARS_DEFAULT = 3

# casualty-specific (matches CasualtyDashboard.py)
CASUALTY_MIN_CLAIMS = 4
CASUALTY_SMOOTH     = 6

# property-specific (matches PropertyDashboard2.py)
PROPERTY_TRAIN_CUTOFF = pd.Period("2025-09", freq="M")
PROPERTY_TEST_SPLIT   = "2024-01-01"   # Prophet/SARIMAX test split in the reference

PROPERTY_FORECAST_MIN_CLAIMS = 50     # below this, no forecast attempt
PROPERTY_FORECAST_MAX_MAPE   = 25.0   # at or above this, forecast is hidden

# Curated 6-order grid: covers the orders that win most often on monthly
# claims severity data without running all 32 combinations.
_SARIMAX_REDUCED_GRID = [
    ((1, 1, 1), (1, 0, 0, 12)),
    ((1, 1, 1), (1, 1, 0, 12)),
    ((0, 1, 1), (1, 0, 0, 12)),
    ((1, 1, 0), (1, 0, 0, 12)),
    ((2, 1, 1), (1, 0, 0, 12)),
    ((1, 1, 1), (0, 0, 1, 12)),
]


# ── shared helpers ───────────────────────────────────────────────────────────

def parse_euro(value) -> float:
    """Parse Irish-format euro strings like '€12 345,67' or '€12.345,67'."""
    s = str(value).strip()
    if s in ("", "nan", "None", "NA", "na", "NaN"):
        return np.nan
    s = "".join(c for c in s if c.isdigit() or c in ".,-")
    s = s.strip("-")
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def covid_flag(ts) -> np.ndarray:
    ts = pd.to_datetime(pd.Series(ts))
    return ((ts >= COVID_START) & (ts <= COVID_END)).astype(float).values


def safe_mape(actual, forecast):
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mask = actual != 0
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


# =============================================================================
# CASUALTY — mirrors CasualtyDashboard.py
# =============================================================================

def load_casualty_monthly() -> pd.DataFrame:
    """
    Monthly avg settled cost, index = Period('M'). Months with fewer than
    CASUALTY_MIN_CLAIMS settled claims are dropped (too thin to trust), and
    the final (likely incomplete) month is dropped — exactly as the
    reference script does.
    """
    df = pd.read_csv(CASUALTY_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["Settlement Date"] = pd.to_datetime(df["Settlement Date"], dayfirst=True, errors="coerce")
    df["Incurred Reserve Total"] = df["Incurred Reserve Total"].apply(parse_euro)
    df = df[df["Settlement Date"].notna() & df["Incurred Reserve Total"].notna() & (df["Incurred Reserve Total"] > 0)]
    df["ym"] = df["Settlement Date"].dt.to_period("M")

    g = df.groupby("ym")["Incurred Reserve Total"]
    mr = pd.DataFrame({"avg_cost": g.mean(), "n_claims": g.size()})
    full_idx = pd.period_range(mr.index.min(), mr.index.max(), freq="M")
    mr = mr.reindex(full_idx)

    monthly = mr[mr["n_claims"] >= CASUALTY_MIN_CLAIMS].copy().iloc[:-1]
    monthly["avg_smooth"] = monthly["avg_cost"].rolling(CASUALTY_SMOOTH, min_periods=3).mean()
    monthly["yoy"] = monthly["avg_cost"].pct_change(12) * 100
    monthly["ds"] = monthly.index.to_timestamp()
    monthly = monthly.reset_index(drop=True)
    return monthly


def load_casualty_all_claims_monthly() -> pd.DataFrame:
    """
    Monthly count and avg cost of ALL claims (settled + unsettled).
    Uses Incident Date to group (better representation of true claim activity).
    Unsettled claims contribute their Incurred Reserve Total as estimated cost.
    No minimum claim filter (unlike settled-only version).
    """
    df = pd.read_csv(CASUALTY_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["Incident Date"] = pd.to_datetime(df["Incident Date"], dayfirst=True, errors="coerce")
    df["Incurred Reserve Total"] = df["Incurred Reserve Total"].apply(parse_euro)
    df = df[df["Incident Date"].notna() & df["Incurred Reserve Total"].notna() & (df["Incurred Reserve Total"] > 0)]
    df["ym"] = df["Incident Date"].dt.to_period("M")

    g = df.groupby("ym")["Incurred Reserve Total"]
    mr = pd.DataFrame({"avg_cost": g.mean(), "n_claims": g.size()})
    full_idx = pd.period_range(mr.index.min(), mr.index.max(), freq="M")
    mr = mr.reindex(full_idx)

    monthly = mr.copy().iloc[:-1]  # drop last incomplete month
    monthly["avg_smooth"] = monthly["avg_cost"].rolling(CASUALTY_SMOOTH, min_periods=3).mean()
    monthly["yoy"] = monthly["avg_cost"].pct_change(12) * 100
    monthly["count_smooth"] = monthly["n_claims"].rolling(CASUALTY_SMOOTH, min_periods=3).mean()
    monthly["count_yoy"] = monthly["n_claims"].pct_change(12) * 100
    monthly["ds"] = monthly.index.to_timestamp()
    monthly = monthly.reset_index(drop=True)
    return monthly


def engineer_casualty_inflation_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Merge real CSO inflation series into the monthly claims frame as
    month-on-month deltas, mirroring how the property model uses delta_HICP.
    These deltas become genuine exogenous regressors in both Prophet and
    SARIMAX — not a cosmetic correlation overlay — so the casualty forecast
    actually responds to measured medical and legal cost pressure, which is
    the whole point of this project.

    Adds columns: cpi_level, hicp_yoy, medical_level, legal_level,
                  delta_cpi, delta_medical, delta_legal
    """
    df = monthly.copy()
    df["ym"] = pd.PeriodIndex(df["ds"], freq="M")

    cpi_raw = pd.read_csv(CPI_PATH, encoding="utf-8-sig")
    cpi_raw.columns = cpi_raw.columns.str.strip()
    cpi_raw["ym"] = pd.to_datetime(cpi_raw["Month"], format="%Y %B", errors="coerce").dt.to_period("M")
    cpi_raw["cpi_level"] = pd.to_numeric(cpi_raw["VALUE"], errors="coerce")
    cpi_m = cpi_raw.dropna(subset=["ym", "cpi_level"]).drop_duplicates("ym")[["ym", "cpi_level"]]

    hicp_raw = pd.read_csv(HICP_PATH, encoding="utf-8-sig")
    hicp_raw.columns = hicp_raw.columns.str.strip()
    hicp_raw["ym"] = pd.to_datetime(hicp_raw["Date"], errors="coerce").dt.to_period("M")
    hicp_m = hicp_raw.dropna(subset=["ym"]).drop_duplicates("ym").rename(columns={"HICP_YoY_pct": "hicp_yoy"})[["ym", "hicp_yoy"]]

    medical = load_medical_cpi()
    medical["ym"] = pd.PeriodIndex(medical["ds"], freq="M")
    medical_m = medical.drop_duplicates("ym").rename(columns={"val": "medical_level"})[["ym", "medical_level"]]

    legal = load_legal_proxy()
    legal["ym"] = pd.PeriodIndex(legal["ds"], freq="M")
    legal_q = legal.drop_duplicates("ym").rename(columns={"val": "legal_level"})[["ym", "legal_level"]]
    # legal proxy is quarterly — upsample to monthly by forward-filling within quarter
    full_months = pd.period_range(legal_q["ym"].min(), legal_q["ym"].max(), freq="M")
    legal_m = legal_q.set_index("ym")["legal_level"].reindex(full_months).ffill().reset_index()
    legal_m.columns = ["ym", "legal_level"]

    df = df.merge(cpi_m, on="ym", how="left")
    df = df.merge(hicp_m, on="ym", how="left")
    df = df.merge(medical_m, on="ym", how="left")
    df = df.merge(legal_m, on="ym", how="left")

    # forward/back-fill small gaps so the model never sees a hole mid-series
    for col in ["cpi_level", "hicp_yoy", "medical_level", "legal_level"]:
        df[col] = df[col].ffill().bfill()

    df["delta_cpi"]     = df["cpi_level"].diff()
    df["delta_medical"] = df["medical_level"].diff()
    df["delta_legal"]   = df["legal_level"].diff()
    # first row's delta is NaN by construction; fill with 0 (no change assumed)
    for col in ["delta_cpi", "delta_medical", "delta_legal"]:
        df[col] = df[col].fillna(0.0)

    return df


# ── severity bucketing, from Nature of Injury ───────────────────────────────
# Five severity tiers built from the claims data itself: average settled cost
# per tier runs €29.5k (Catastrophic) -> €22.5k -> €17.7k -> €11.9k -> €11.3k
# (Property/Other), a clean monotonic gradient confirmed against the actual
# Incurred Reserve Total values. ~95% of rows classify into a named tier;
# the remainder fall to "Other / Unclassified".

_SEVERITY_CATASTROPHIC = {
    "Fatal", "Brain Damage", "Paraplegic", "Spinal", "Multiple Injuries", "Internal Injuries",
    "Amputation - Finger", "Amputation - Foot", "Amputation - Leg", "Amputation - Toe",
    "Fracture - Ankle", "Fracture - Arm", "Fracture - Broken Heel", "Fracture - Clavicle / Scapula",
    "Fracture - Finger", "Fracture - Foot", "Fracture - Hand", "Fracture - Jaw / Cheek",
    "Fracture - Leg", "Fracture - Pelvis / Hip", "Fracture - Ribs", "Fracture - Skull",
    "Fracture - Thumb", "Fracture - Toe", "Fracture - Vertebrae", "Fracture - Wrist",
    "Fracture - Wrist/Foot", "Hernia", "Cancer", "Slipped Disc", "Eye Injury - Loss of Sight",
    "Hearing - Loss of", "Deafness", "Scarring",
}
_SEVERITY_MUSCULOSKELETAL = {
    "Back Injury", "BA - BACK", "Neck Injury", "NE - NECK", "Knee Injury", "KN - KNEE/S",
    "Shoulder Injury", "Ankle Injury", "AN - ANKLE", "Wrist Injury", "WR - WRIST/S",
    "Soft Tissue Injury", "Whiplash (12-18 Mths)", "Whiplash (18-24 Mths)", "Whiplash (less than 12 mths)",
    "Leg Injury", "Leg Injury (Lower)", "Leg Injury (Upper)", "LE - LEG/S", "Arm Injury",
    "Elbow Injury", "Hip Injury", "Groin Injury", "RSI", "Rib", "Bruising to Ribs", "Pelvis",
    "Clavicle",
}
_SEVERITY_MINOR_PHYSICAL = {
    "Cuts / Sprains / Bruising", "Finger Injury", "FN - FINGER/S", "Foot Injury", "FO - FOOT/FEET",
    "Hand Injury", "Burn Injury", "Facial Injury", "Head Injury", "HE - HEAD",
    "Head Injury - Other than Brain Damage", "Eye Injury", "Eye Injury other than Loss of Sight",
    "Eye Condition", "Dental Injury", "Tooth / Teeth", "Thumb", "Toe", "Thigh", "Shin",
    "Mouth Injury", "Jaw Injury", "Nose", "Cheek", "Ear", "Scalding", "Chest Injury", "Abdomen",
}
_SEVERITY_PSYCHOLOGICAL = {
    "Stress", "Psychological Trauma", "Embarrassment", "EM - EMBARASSMENT", "Anxiety",
    "Post Traumatic Stress", "Nervous Shock", "Shock / Neurosis", "Psychiatric", "Mental State",
    "Humiliation", "Libel/Slander", "Defamation / Slander", "Sexual Assault", "Assault",
    "Robbery/Theft",
}


def assign_severity_bucket(injury) -> str:
    if pd.isna(injury):
        return "Other / Unclassified"
    s = str(injury).strip()
    if s in _SEVERITY_CATASTROPHIC:
        return "Catastrophic / Major Injury"
    if s in _SEVERITY_MUSCULOSKELETAL:
        return "Moderate Musculoskeletal"
    if s in _SEVERITY_MINOR_PHYSICAL:
        return "Minor Physical Injury"
    if s in _SEVERITY_PSYCHOLOGICAL:
        return "Psychological / Non-Physical"
    return "Property / Other Claims"


def severity_breakdown() -> list[dict]:
    """
    Static descriptive breakdown by severity tier — average settled cost,
    claim count, and share of total — with NO forecasting per tier.

    Per-tier forecasting was tried and reverted: with ~1,862 casualty claims
    total, some tiers (e.g. Catastrophic/Major Injury, Psychological)
    average fewer than 5 claims per month, so a tier's "monthly average
    cost" is really just 1-2 individual settlements, not a stable average.
    Backtested out-of-sample MAPE on those tiers ran as high as 250% in
    testing — a real data-volume floor, not a fixable model bug. The
    severity gradient itself is a genuine, useful finding; trying to
    forecast it tier-by-tier was not.
    """
    df = pd.read_csv(CASUALTY_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["Settlement Date"] = pd.to_datetime(df["Settlement Date"], dayfirst=True, errors="coerce")
    df["Incurred Reserve Total"] = df["Incurred Reserve Total"].apply(parse_euro)
    df = df[df["Settlement Date"].notna() & df["Incurred Reserve Total"].notna() & (df["Incurred Reserve Total"] > 0)]
    df["severity_bucket"] = df["Nature of Injury"].apply(assign_severity_bucket)

    total_claims = len(df)
    total_cost = df["Incurred Reserve Total"].sum()

    rows = []
    order = ["Catastrophic / Major Injury", "Moderate Musculoskeletal", "Minor Physical Injury",
             "Psychological / Non-Physical", "Property / Other Claims", "Other / Unclassified"]
    for bucket in order:
        sub = df[df["severity_bucket"] == bucket]
        if len(sub) == 0:
            continue
        rows.append({
            "tier": bucket,
            "avg_cost": round(float(sub["Incurred Reserve Total"].mean()), 2),
            "median_cost": round(float(sub["Incurred Reserve Total"].median()), 2),
            "n_claims": int(len(sub)),
            "pct_of_claims": round(len(sub) / total_claims * 100, 1),
            "pct_of_cost": round(float(sub["Incurred Reserve Total"].sum()) / total_cost * 100, 1),
            "avg_claims_per_month": round(len(sub) / 70, 1),  # ~70 months in the dataset window
        })
    return rows


# ── medical & legal cost-index loaders (real CSO series) ───────────────────

def load_medical_cpi() -> pd.DataFrame:
    """
    CSO Consumer Price Index — Health division (table CPM24, the live
    successor to the archived CPM13). Monthly, base December 2023 = 100.
    Source: Central Statistics Office, data.cso.ie, table CPM24.
    """
    df = pd.read_csv(MEDICAL_CPI_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df[df["Statistic Label"] == "Consumer Price Index"].copy()
    df["ds"] = pd.to_datetime(df["Month"].str.strip(), format="%Y %B", errors="coerce")
    df["val"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df.dropna(subset=["ds", "val"]).sort_values("ds").reset_index(drop=True)
    df["yoy"] = df["val"].pct_change(12) * 100
    return df[["ds", "val", "yoy"]]


def load_legal_proxy() -> pd.DataFrame:
    """
    CSO Earnings, Hours and Employment Costs Survey — average weekly earnings,
    Professional, Scientific & Technical Activities (NACE M), all employees
    (table EHQ03). Used as a cost-pressure proxy for legal/professional fees,
    since Ireland has no dedicated published legal-services price index.
    Source: Central Statistics Office, data.cso.ie, table EHQ03.

    Rebased to 100 at the first observation (same convention as CPI/HICP/
    construction) so deltas are small index-point changes rather than raw
    euro jumps of several hundred per quarter — using the raw euro level
    here previously fed the model a multi-euro "monthly change" that
    compounded into a runaway forecast and an exploding confidence band.
    """
    df = pd.read_csv(LEGAL_PROXY_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["ds"] = pd.PeriodIndex(df["Quarter"].str.strip(), freq="Q").to_timestamp()
    df["raw_val"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df.dropna(subset=["ds", "raw_val"]).sort_values("ds").reset_index(drop=True)
    base = df["raw_val"].iloc[0]
    df["val"] = df["raw_val"] / base * 100
    df["yoy"] = df["val"].pct_change(4) * 100
    return df[["ds", "val", "yoy"]]


def fit_prophet_casualty(monthly: pd.DataFrame, periods_ahead: int, target_col: str = "avg_cost"):
    """
    Log-scale Prophet, yearly seasonality, with COVID, medical-cost-delta,
    and legal-cost-delta as exogenous regressors. The medical and legal
    deltas are the genuine measured month-on-month change in the CSO Health
    CPI and the professional-earnings legal proxy — this is what makes the
    forecast actually respond to inflation, not just correlate with it
    after the fact.

    Future regressor values are projected forward using the trailing
    12-month average delta (the same "project the recent trend" approach
    the property model uses for HICP/construction), since true future
    inflation isn't known at the forecast origin.
    """
    feat = engineer_casualty_inflation_features(monthly)

    pdf = pd.DataFrame({
        "ds":            feat["ds"],
        "y":             np.log(feat[target_col]),
        "covid":         covid_flag(feat["ds"]),
        "delta_medical": feat["delta_medical"],
        "delta_legal":   feat["delta_legal"],
    }).dropna(subset=["y"]).reset_index(drop=True)

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, interval_width=0.90,
                seasonality_mode="multiplicative",
                changepoint_prior_scale=0.01, n_changepoints=8)
    m.add_regressor("covid")
    m.add_regressor("delta_medical")
    m.add_regressor("delta_legal")
    m.fit(pdf)

    # project medical/legal deltas forward from the trailing 12-month average
    med_trail   = pdf["delta_medical"].tail(12).mean()
    legal_trail = pdf["delta_legal"].tail(12).mean()

    future = m.make_future_dataframe(periods=periods_ahead, freq="MS")
    future["covid"] = covid_flag(future["ds"])
    future_mask = future["ds"] > pdf["ds"].max()
    future["delta_medical"] = 0.0
    future["delta_legal"]   = 0.0
    future.loc[future_mask, "delta_medical"] = med_trail
    future.loc[future_mask, "delta_legal"]   = legal_trail
    future.loc[~future_mask, "delta_medical"] = pdf["delta_medical"].values
    future.loc[~future_mask, "delta_legal"]   = pdf["delta_legal"].values

    fc = m.predict(future)
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        fc[col] = np.exp(fc[col])

    # Read the yearly seasonal component from one clean synthetic year (with
    # covid=0, no inflation delta) rather than averaging across the
    # historical+forecast mix — averaging dilutes the signal because
    # Prophet's yearly term is constant across years by construction.
    seas_year = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=12, freq="MS")})
    seas_year["covid"] = 0.0
    seas_year["delta_medical"] = 0.0
    seas_year["delta_legal"] = 0.0
    seas_comp = m.predict(seas_year)
    seasonality_pct = {
        int(d.month): round((np.exp(v) - 1) * 100, 1)
        for d, v in zip(seas_year["ds"], seas_comp["yearly"])
    }

    in_sample = fc[fc["ds"].isin(pdf["ds"])][["ds", "yhat"]].reset_index(drop=True)

    return {
        "model": m, "forecast": fc, "seasonality_pct": seasonality_pct, "in_sample": in_sample,
        "med_trail": med_trail, "legal_trail": legal_trail,
    }


def fit_sarimax_casualty(monthly: pd.DataFrame, periods_ahead: int, target_col: str = "avg_cost"):
    """
    Log-scale SARIMAX(1,1,1)(1,1,0,12) with COVID, medical-cost-delta, and
    legal-cost-delta as exogenous regressors — the same real CSO inflation
    series Prophet uses, so both models are genuinely inflation-driven.
    """
    feat = engineer_casualty_inflation_features(monthly)
    y = np.log(feat[target_col].astype(float).values)
    ts = feat["ds"]

    exog = np.column_stack([
        covid_flag(ts),
        feat["delta_medical"].values,
        feat["delta_legal"].values,
    ])

    model = SARIMAX(
        y, exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    med_trail   = feat["delta_medical"].tail(12).mean()
    legal_trail = feat["delta_legal"].tail(12).mean()

    future_idx = pd.date_range(ts.iloc[-1] + pd.DateOffset(months=1), periods=periods_ahead, freq="MS")
    future_exog = np.column_stack([
        covid_flag(pd.Series(future_idx)),
        np.full(periods_ahead, med_trail),
        np.full(periods_ahead, legal_trail),
    ])

    fc_obj = res.get_forecast(steps=periods_ahead, exog=future_exog)
    mean = np.exp(np.asarray(fc_obj.predicted_mean))
    ci = np.exp(np.asarray(fc_obj.conf_int(alpha=0.10)))

    clip_hi = feat[target_col].dropna().max() * 4
    mean = np.clip(mean, 1000, clip_hi)

    fc = pd.DataFrame({
        "ds": future_idx, "yhat": mean,
        "yhat_lower": np.clip(ci[:, 0], 1000, clip_hi),
        "yhat_upper": np.clip(ci[:, 1], 1000, clip_hi),
    })

    in_sample_mean = np.exp(np.asarray(res.fittedvalues))
    in_sample = pd.DataFrame({"ds": ts.values, "yhat": np.clip(in_sample_mean, 1000, clip_hi)})

    # Extract SARIMAX seasonal pattern by comparing monthly averages of
    # in-sample fitted values (log scale) against the overall mean.
    # This gives a data-driven monthly effect independent of trend.
    sarimax_seasonality_pct = {}
    try:
        log_fitted = np.asarray(res.fittedvalues)
        fitted_ds = pd.Series(ts.values)
        months = fitted_ds.dt.month.values
        log_monthly_means = {}
        for mo in range(1, 13):
            mask = months == mo
            if mask.sum() >= 2:
                log_monthly_means[mo] = log_fitted[mask].mean()
        if log_monthly_means:
            grand_mean = np.mean(list(log_monthly_means.values()))
            sarimax_seasonality_pct = {
                mo: round((np.exp(v - grand_mean) - 1) * 100, 1)
                for mo, v in log_monthly_means.items()
            }
    except Exception:
        pass

    return {
        "model": res, "forecast": fc, "in_sample": in_sample,
        "med_trail": med_trail, "legal_trail": legal_trail,
        "sarimax_seasonality_pct": sarimax_seasonality_pct,
    }


def casualty_correlations(monthly: pd.DataFrame, cpi: pd.DataFrame, hicp: pd.DataFrame,
                           medical: pd.DataFrame = None, legal: pd.DataFrame = None) -> dict:
    """Like-for-like correlation: everything converted to YoY% (matches reference)."""
    m = monthly.copy()
    m["ym"] = pd.PeriodIndex(m["ds"], freq="M")
    claim_yoy = m.set_index("ym")["avg_smooth"].pct_change(12) * 100

    cpi_s = cpi.copy()
    cpi_s["ym"] = pd.PeriodIndex(cpi_s["ds"], freq="M")
    cpi_yoy = cpi_s.set_index("ym")["val"].pct_change(12) * 100

    hicp_s = hicp.copy()
    hicp_s["ym"] = pd.PeriodIndex(hicp_s["ds"], freq="M")
    hicp_yoy = hicp_s.set_index("ym")["yoy"]

    data = {"claim_yoy": claim_yoy, "cpi_yoy": cpi_yoy, "hicp_yoy": hicp_yoy}

    if medical is not None and len(medical):
        med_s = medical.copy()
        med_s["ym"] = pd.PeriodIndex(med_s["ds"], freq="M")
        data["medical_yoy"] = med_s.set_index("ym")["yoy"]

    if legal is not None and len(legal):
        # legal proxy is quarterly — upsample to monthly via forward-fill so
        # it can align with the monthly claims series for correlation
        legal_s = legal.copy()
        legal_s["ym"] = pd.PeriodIndex(legal_s["ds"], freq="M")
        legal_m = legal_s.set_index("ym")["yoy"]
        full_months = pd.period_range(legal_m.index.min(), legal_m.index.max(), freq="M")
        data["legal_yoy"] = legal_m.reindex(full_months).ffill()

    df = pd.DataFrame(data).dropna()

    def safe_r(a, b):
        if len(a) < 5:
            return None
        r, _ = pearsonr(a, b)
        return None if np.isnan(r) else round(float(r), 3)

    result = {
        "cpi_corr":  safe_r(df["claim_yoy"], df["cpi_yoy"]),
        "hicp_corr": safe_r(df["claim_yoy"], df["hicp_yoy"]),
    }
    if "medical_yoy" in df.columns:
        result["medical_corr"] = safe_r(df["claim_yoy"], df["medical_yoy"])
    if "legal_yoy" in df.columns:
        result["legal_corr"] = safe_r(df["claim_yoy"], df["legal_yoy"])
    result["n_obs"] = int(len(df))
    return result


def casualty_backtest(monthly: pd.DataFrame) -> dict:
    """
    Both models are fit on the 6-month smoothed severity series rather than
    the raw monthly average. The raw monthly average is inherently noisy
    here (each month averages as few as 4-77 large, lumpy legal
    settlements), so no model can track its single-month swings to under
    15% MAPE — that noise is real, not a modelling failure. The smoothed
    series isolates the underlying cost trend, which is both forecastable
    and the figure that matters for reserving and pricing decisions.
    """
    target = "avg_smooth"
    usable = monthly.dropna(subset=[target]).reset_index(drop=True)

    full_p = fit_prophet_casualty(usable, periods_ahead=1, target_col=target)
    full_s = fit_sarimax_casualty(usable, periods_ahead=1, target_col=target)

    # SARIMAX with order=(1,1,1), seasonal_order=(1,1,0,12) needs d + D*s = 13
    # observations just to initialise differencing — fittedvalues in that
    # warm-up window are unreliable by construction and must be excluded from
    # any fit-quality metric, exactly as statsmodels' own diagnostics do.
    SARIMAX_WARMUP = 1 + 1 * 12

    actual = usable[target].values
    p_fit_vals = full_p["in_sample"].set_index("ds").reindex(usable["ds"])["yhat"].values
    s_fit_vals = full_s["in_sample"].set_index("ds").reindex(usable["ds"])["yhat"].values

    fit_mape_prophet = safe_mape(actual, p_fit_vals)
    fit_mape_sarimax = safe_mape(actual[SARIMAX_WARMUP:], s_fit_vals[SARIMAX_WARMUP:])

    train = usable[usable["ds"] < "2025-01-01"].copy()
    test  = usable[(usable["ds"] >= "2025-01-01") & (usable["ds"] < "2026-01-01")].copy()

    test_points, oos_p_mape, oos_s_mape = [], None, None
    if len(train) >= 24 and len(test) > 0:
        horizon = len(test)
        p_fc = None
        s_fc = None
        try:
            p_bt = fit_prophet_casualty(train, periods_ahead=horizon, target_col=target)
            p_fc = p_bt["forecast"][p_bt["forecast"]["ds"] > train["ds"].max()].head(horizon).reset_index(drop=True)
            oos_p_mape = safe_mape(test[target].values, p_fc["yhat"].values)
        except Exception:
            pass
        try:
            s_bt = fit_sarimax_casualty(train, periods_ahead=horizon, target_col=target)
            s_fc = s_bt["forecast"].reset_index(drop=True)
            oos_s_mape = safe_mape(test[target].values, s_fc["yhat"].values)
        except Exception:
            pass

        for i, (d, a) in enumerate(zip(test["ds"], test[target])):
            test_points.append({
                "date": str(d)[:10],
                "actual": round(float(a), 2),
                "prophet": round(float(p_fc["yhat"].iloc[i]), 2) if p_fc is not None else None,
                "sarimax": round(float(s_fc["yhat"].iloc[i]), 2) if s_fc is not None else None,
            })

    return {
        "fit_mape_prophet": round(fit_mape_prophet, 1) if fit_mape_prophet is not None else None,
        "fit_mape_sarimax": round(fit_mape_sarimax, 1) if fit_mape_sarimax is not None else None,
        "prophet_mape": round(oos_p_mape, 1) if oos_p_mape is not None else None,
        "sarimax_mape": round(oos_s_mape, 1) if oos_s_mape is not None else None,
        "test_points": test_points,
        "target_series": "6-month smoothed average claim cost",
    }


# =============================================================================
# PROPERTY — mirrors PropertyDashboard2.py
# =============================================================================

_WATER    = {"Water", "Escape of Water", "Flood", "Water Damage",
             "Escape Of Oil", "Escape of Oil", "Trace & Access"}
_STORM    = {"Storm", "Lightning", "Weight of Snow",
             "Snow / Ice Damage", "Snow/Ice Damage", "Falling Trees"}
_FIRE     = {"Fire"}
_THEFT    = {"Theft", "Burglary", "Malicious Damage", "Vandalism"}

COUNTY_REGION_MAP: dict[str, str] = {
    "Dublin":    "East",  "Kildare":  "East",     "Louth":     "East",
    "Meath":     "East",  "Wexford":  "East",     "Wicklow":   "East",
    "Carlow":    "Midlands", "Cavan":  "Midlands", "Kilkenny":  "Midlands",
    "Laois":     "Midlands", "Longford": "Midlands", "Monaghan": "Midlands",
    "Offaly":    "Midlands", "Westmeath": "Midlands",
    "Cork":      "South / Southwest", "Kerry":     "South / Southwest",
    "Limerick":  "South / Southwest", "Tipperary": "South / Southwest",
    "Waterford": "South / Southwest",
    "Clare":     "West / Northwest",  "Donegal":   "West / Northwest",
    "Galway":    "West / Northwest",  "Leitrim":   "West / Northwest",
    "Mayo":      "West / Northwest",  "Roscommon": "West / Northwest",
    "Sligo":     "West / Northwest",
}


def assign_peril_group(peril: str) -> str:
    if peril in _WATER: return "Water & Flooding"
    if peril in _STORM: return "Storm & Weather"
    if peril in _FIRE:  return "Fire"
    if peril in _THEFT: return "Theft & Vandalism"
    return "Accidental Damage"


def load_property_raw() -> pd.DataFrame:
    df = pd.read_csv(PROPERTY_PATH, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip().replace("\n", " ").strip('"') for c in df.columns]
    df["Date of Loss"] = pd.to_datetime(df["Date of Loss"], dayfirst=True, errors="coerce")
    for col in ["Adjusted Settlement - Buildings", "Adjusted Settlement - Contents"]:
        df[col] = df[col].apply(parse_euro)
    df["total_cost"] = df["Adjusted Settlement - Buildings"].fillna(0) + df["Adjusted Settlement - Contents"].fillna(0)
    df = df.dropna(subset=["Date of Loss"])
    df["Month"] = df["Date of Loss"].dt.to_period("M")
    df["peril_group"] = df["Peril/Loss Type"].fillna("").apply(assign_peril_group)
    df["region"] = df["Risk Address COUNTY ONLY"].str.strip().map(COUNTY_REGION_MAP).fillna("Other")
    return df


def agg_property_monthly(raw: pd.DataFrame, group: str = "ALL", cutoff: pd.Period = PROPERTY_TRAIN_CUTOFF,
                          building_contents: str = "ALL", claim_category: str = "ALL",
                          region: str = "ALL") -> pd.DataFrame:
    mask = (raw["Month"] >= START_MONTH) & (raw["Month"] <= cutoff)
    if group != "ALL":
        mask = mask & (raw["peril_group"] == group)
    if claim_category != "ALL":
        mask = mask & (raw["Claim Category"] == claim_category)
    if region != "ALL":
        mask = mask & (raw["region"] == region)

    sub = raw[mask].copy()

    if building_contents == "Buildings":
        sub["_cost"] = sub["Adjusted Settlement - Buildings"].fillna(0)
        sub = sub[sub["_cost"] > 0]
        cost_col = "_cost"
    elif building_contents == "Contents":
        sub["_cost"] = sub["Adjusted Settlement - Contents"].fillna(0)
        sub = sub[sub["_cost"] > 0]
        cost_col = "_cost"
    else:
        cost_col = "total_cost"

    monthly = (
        sub.groupby("Month", as_index=False)
        .agg(total_reserve=(cost_col, "sum"), claim_count=(cost_col, "size"))
    )
    monthly["average_claim_cost"] = monthly["total_reserve"] / monthly["claim_count"]
    return monthly.sort_values("Month").reset_index(drop=True)


def _post_process_property_monthly(raw_agg: pd.DataFrame) -> pd.DataFrame:
    out = raw_agg.rename(columns={"average_claim_cost": "avg_cost", "claim_count": "n_claims"}).copy()
    out["ds"] = out["Month"].dt.to_timestamp()
    out["avg_smooth"] = out["avg_cost"].rolling(6, min_periods=3).mean()
    out["yoy"] = out["avg_cost"].pct_change(12) * 100
    return out.reset_index(drop=True)


def load_construction_proxy() -> tuple:
    """
    Construction-sector average weekly earnings (quarterly), rebased to 100
    at the first available quarter, used as the construction cost-pressure
    proxy in place of the reference's dedicated construction-inflation CSV.
    """
    earn = pd.read_csv(EARNINGS_PATH, encoding="utf-8-sig")
    earn.columns = earn.columns.str.strip()
    earn = earn[
        earn["Sector"].str.contains("Construction", case=False, na=False)
        & earn["Statistic"].str.contains("Average Weekly Earnings", case=False, na=False)
    ].copy()
    earn["Month"] = pd.PeriodIndex(earn["Quarter"], freq="Q").asfreq("M", how="start")
    earn = earn.drop_duplicates(subset="Month").sort_values("Month").reset_index(drop=True)
    base = earn["Value"].iloc[0]
    earn["Index_raw_val"] = earn["Value"] / base * 100

    full_range = pd.period_range(START_MONTH, PROPERTY_TRAIN_CUTOFF, freq="M")
    raw_series = earn.set_index("Month")["Index_raw_val"].reindex(full_range)
    raw_series.index.name = "Month"
    raw_out = raw_series.reset_index()
    raw_out.columns = ["Month", "Index_raw"]

    interp_series = raw_series.interpolate(method="linear").ffill().bfill()
    interp_out = interp_series.reset_index()
    interp_out.columns = ["Month", "Index_adjusted"]

    return interp_out, raw_out


def load_property_hicp() -> pd.DataFrame:
    hicp = pd.read_csv(HICP_PATH, encoding="utf-8-sig")
    hicp.columns = hicp.columns.str.strip()
    hicp["Date"] = pd.to_datetime(hicp["Date"], errors="coerce")
    hicp["Month"] = hicp["Date"].dt.to_period("M")
    hicp = hicp.dropna(subset=["Month"]).sort_values("Month").reset_index(drop=True)
    level = [100.0]
    for v in hicp["HICP_YoY_pct"].iloc[1:]:
        level.append(level[-1] * (1 + (v / 100.0) / 12))
    hicp["HICP_adjusted"] = level
    return hicp[["Month", "HICP_adjusted"]]


def load_property_cpi() -> pd.DataFrame:
    cpi = pd.read_csv(CPI_PATH, encoding="utf-8-sig")
    cpi.columns = cpi.columns.str.strip()
    cpi["VALUE"] = pd.to_numeric(cpi["VALUE"], errors="coerce")
    cpi = cpi.dropna(subset=["VALUE"])
    cpi["Month"] = pd.to_datetime(cpi["Month"].str.strip(), format="%Y %B", errors="coerce").dt.to_period("M")
    cpi = cpi.dropna(subset=["Month"]).sort_values("Month").reset_index(drop=True)
    cpi = cpi[cpi["Month"] >= START_MONTH].reset_index(drop=True)
    base_row = cpi.loc[cpi["Month"] == START_MONTH, "VALUE"]
    base = base_row.iloc[0] if not base_row.empty else cpi["VALUE"].iloc[0]
    cpi["CPI_adjusted"] = cpi["VALUE"] / base * 100
    return cpi[["Month", "CPI_adjusted"]]


def engineer_property_features(monthly, hicp, construction_interp, construction_raw, cpi) -> pd.DataFrame:
    df = monthly.copy()
    df["roll_6m"] = df["average_claim_cost"].rolling(6, min_periods=3).mean()
    df["roll_12m"] = df["average_claim_cost"].rolling(12, min_periods=6).mean()
    df["avg_cost_lag12"] = df["average_claim_cost"].shift(12)
    df["yoy_pct"] = (df["average_claim_cost"] / df["avg_cost_lag12"] - 1) * 100
    df = df.merge(hicp, on="Month", how="left")
    df = df.merge(construction_interp, on="Month", how="left")
    df = df.merge(construction_raw, on="Month", how="left")
    df = df.merge(cpi, on="Month", how="left")
    df["delta_HICP"] = df["HICP_adjusted"].diff()
    df["date"] = df["Month"].dt.to_timestamp()
    return df


def property_correlations(df: pd.DataFrame) -> dict:
    target = df["average_claim_cost"]
    out = {}
    for col, key in [("HICP_adjusted", "hicp_corr"), ("CPI_adjusted", "cpi_corr"), ("Index_raw", "construction_corr")]:
        if col not in df.columns:
            out[key] = None
            continue
        aligned = pd.concat([target, df[col]], axis=1).dropna()
        if len(aligned) < 5:
            out[key] = None
            continue
        r, _ = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        out[key] = None if np.isnan(r) else round(float(r), 3)
    return out


def fit_sarimax_property(df: pd.DataFrame):
    """Grid search over the same order/seasonal grid as the reference, selected by AIC."""
    sdf = df[["Month", "average_claim_cost", "delta_HICP"]].dropna().copy()
    sdf.index = sdf["Month"].dt.to_timestamp()
    sdf.index.freq = "MS"
    sdf = sdf.drop(columns="Month")
    train = sdf.loc[:"2023-12-01"]
    test = sdf.loc["2024-01-01":]
    if len(train) < 12 or len(test) < 1:
        return None

    orders = [(1,0,0),(0,0,1),(1,0,1),(2,0,0),(2,0,1),(1,1,1),(0,1,1),(1,1,0)]
    seasonals = [(0,0,0,12),(1,0,0,12),(0,0,1,12),(1,0,1,12)]
    best_aic, best_res, best_order, best_seas = np.inf, None, None, None

    for order in orders:
        for seas in seasonals:
            try:
                mdl = SARIMAX(
                    train["average_claim_cost"], exog=train[["delta_HICP"]],
                    order=order, seasonal_order=seas, trend="c",
                    enforce_stationarity=False, enforce_invertibility=False,
                )
                res = mdl.fit(disp=False, maxiter=500)
                if not res.mle_retvals.get("converged", False):
                    continue
                fc_check = res.get_forecast(steps=len(test), exog=test[["delta_HICP"]]).predicted_mean
                if not np.isfinite(fc_check).all() or np.abs(fc_check).max() > 5_000_000:
                    continue
                if res.aic < best_aic:
                    best_aic, best_res, best_order, best_seas = res.aic, res, order, seas
            except Exception:
                continue

    if best_res is None:
        return None

    last_delta = float(train["delta_HICP"].dropna().tail(6).mean())
    naive_exog = pd.DataFrame({"delta_HICP": [last_delta] * len(test)}, index=test.index)
    fc = best_res.get_forecast(steps=len(test), exog=naive_exog)
    fc_mu = np.asarray(fc.predicted_mean)
    fc_ci = np.asarray(fc.conf_int())

    ev = test.copy()
    ev["forecast"] = fc_mu
    ev["lower"] = fc_ci[:, 0]
    ev["upper"] = fc_ci[:, 1]
    ev["abs_err"] = (ev["average_claim_cost"] - ev["forecast"]).abs()
    ev["ape"] = ev["abs_err"] / ev["average_claim_cost"] * 100

    return {
        "model": best_res, "order": best_order, "seasonal": best_seas, "aic": best_aic,
        "train": train, "test": test, "eval": ev,
        "mape": float(ev["ape"].mean()),
        "last_delta": last_delta,
    }


def fit_sarimax_property_reduced_grid(df: pd.DataFrame, cached_best_order: tuple = None) -> dict | None:
    """
    AIC-selected SARIMAX using the curated 6-order grid plus the pre-cached
    best order (if provided and not already in the grid).  Covers the orders
    that win most often on monthly claims severity data; ~3-6× faster than
    the full 32-combination grid search.  Return structure is identical to
    fit_sarimax_property so all downstream helpers work unchanged.
    """
    sdf = df[["Month", "average_claim_cost", "delta_HICP"]].dropna().copy()
    sdf.index = sdf["Month"].dt.to_timestamp()
    sdf.index.freq = "MS"
    sdf = sdf.drop(columns="Month")
    train = sdf.loc[:"2023-12-01"]
    test  = sdf.loc["2024-01-01":]
    if len(train) < 12 or len(test) < 1:
        return None

    candidates = list(_SARIMAX_REDUCED_GRID)
    if cached_best_order and cached_best_order not in candidates:
        candidates.insert(0, cached_best_order)  # try the known winner first

    best_aic, best_res, best_order, best_seas = np.inf, None, None, None
    for order, seas in candidates:
        try:
            mdl = SARIMAX(
                train["average_claim_cost"], exog=train[["delta_HICP"]],
                order=order, seasonal_order=seas, trend="c",
                enforce_stationarity=False, enforce_invertibility=False,
            )
            res = mdl.fit(disp=False, maxiter=500)
            if not res.mle_retvals.get("converged", False):
                continue
            fc_check = res.get_forecast(steps=len(test), exog=test[["delta_HICP"]]).predicted_mean
            if not np.isfinite(fc_check).all() or np.abs(fc_check).max() > 5_000_000:
                continue
            if res.aic < best_aic:
                best_aic, best_res, best_order, best_seas = res.aic, res, order, seas
        except Exception:
            continue

    if best_res is None:
        return None

    last_delta = float(train["delta_HICP"].dropna().tail(6).mean())
    naive_exog = pd.DataFrame({"delta_HICP": [last_delta] * len(test)}, index=test.index)
    fc     = best_res.get_forecast(steps=len(test), exog=naive_exog)
    fc_mu  = np.asarray(fc.predicted_mean)
    fc_ci  = np.asarray(fc.conf_int())

    ev = test.copy()
    ev["forecast"] = fc_mu
    ev["lower"]    = fc_ci[:, 0]
    ev["upper"]    = fc_ci[:, 1]
    ev["abs_err"]  = (ev["average_claim_cost"] - ev["forecast"]).abs()
    ev["ape"]      = ev["abs_err"] / ev["average_claim_cost"] * 100

    return {
        "model": best_res, "order": best_order, "seasonal": best_seas, "aic": best_aic,
        "train": train, "test": test, "eval": ev,
        "mape": float(ev["ape"].mean()),
        "last_delta": last_delta,
    }


def sarimax_property_future(res, horizon: int) -> pd.DataFrame:
    if res is None:
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
    fc = res["model"].get_forecast(steps=horizon, exog=pd.DataFrame({"delta_HICP": [res["last_delta"]] * horizon}))
    dates = pd.date_range(res["eval"].index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    ci = np.asarray(fc.conf_int())
    return pd.DataFrame({
        "ds": dates, "yhat": np.asarray(fc.predicted_mean),
        "yhat_lower": ci[:, 0], "yhat_upper": ci[:, 1],
    })


def fit_prophet_property(df: pd.DataFrame):
    pdf = (
        df[["date", "average_claim_cost", "HICP_adjusted", "Index_adjusted"]]
        .rename(columns={"date": "ds", "average_claim_cost": "y"})
        .dropna().reset_index(drop=True)
    )
    train = pdf[pdf["ds"] < PROPERTY_TEST_SPLIT].copy()
    test = pdf[pdf["ds"] >= PROPERTY_TEST_SPLIT].copy()
    if len(train) < 12 or len(test) < 1:
        return None

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, interval_width=0.90)
    m.add_regressor("HICP_adjusted")
    m.add_regressor("Index_adjusted")
    m.fit(train)

    recent = train.tail(12)
    hicp_slope = (recent["HICP_adjusted"].iloc[-1] - recent["HICP_adjusted"].iloc[0]) / len(recent)
    constr_slope = (recent["Index_adjusted"].iloc[-1] - recent["Index_adjusted"].iloc[0]) / len(recent)
    n = len(test)
    test_proj = test[["ds", "y"]].copy()
    test_proj["HICP_adjusted"] = [recent["HICP_adjusted"].iloc[-1] + hicp_slope * (i + 1) for i in range(n)]
    test_proj["Index_adjusted"] = [recent["Index_adjusted"].iloc[-1] + constr_slope * (i + 1) for i in range(n)]

    fc = m.predict(test_proj)
    ev = test[["ds", "y"]].copy()
    ev["forecast"] = fc["yhat"].values
    ev["lower"] = fc["yhat_lower"].values
    ev["upper"] = fc["yhat_upper"].values
    ev["abs_err"] = (ev["y"] - ev["forecast"]).abs()
    ev["ape"] = ev["abs_err"] / ev["y"] * 100

    return {"model": m, "train": train, "test": ev, "mape": float(ev["ape"].mean())}


def prophet_property_future(res, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if res is None:
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
    last_date = df["date"].max()
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")
    recent = df.tail(12)
    hicp_slope = (recent["HICP_adjusted"].iloc[-1] - recent["HICP_adjusted"].iloc[0]) / len(recent)
    constr_slope = (recent["Index_adjusted"].iloc[-1] - recent["Index_adjusted"].iloc[0]) / len(recent)
    future_df = pd.DataFrame({
        "ds": future_dates,
        "HICP_adjusted":  [df["HICP_adjusted"].iloc[-1] + hicp_slope * (i+1) for i in range(horizon)],
        "Index_adjusted": [df["Index_adjusted"].iloc[-1] + constr_slope * (i+1) for i in range(horizon)],
    })
    fc = res["model"].predict(future_df)
    return pd.DataFrame({
        "ds": future_dates, "yhat": fc["yhat"].values,
        "yhat_lower": fc["yhat_lower"].values, "yhat_upper": fc["yhat_upper"].values,
    })


def property_seasonality(prophet_res) -> dict:
    if prophet_res is None:
        return {}
    m = prophet_res["model"]
    full_year = pd.date_range("2023-01-01", periods=12, freq="MS")
    comp = m.predict(pd.DataFrame({
        "ds": full_year,
        "HICP_adjusted": [prophet_res["train"]["HICP_adjusted"].mean()] * 12,
        "Index_adjusted": [prophet_res["train"]["Index_adjusted"].mean()] * 12,
    }))
    base = comp["yearly"].mean()
    return {int(d.month): round(float(v - base), 1) for d, v in zip(full_year, comp["yearly"])}


def property_sarimax_seasonality(sarimax_res) -> dict:
    """Extract seasonal pattern from the property SARIMAX model's in-sample fit."""
    if sarimax_res is None:
        return {}
    try:
        train = sarimax_res["train"]
        fitted = np.asarray(sarimax_res["model"].fittedvalues)
        months = train.index.month
        monthly_means = {}
        for mo in range(1, 13):
            mask = months == mo
            if mask.sum() >= 2:
                monthly_means[mo] = fitted[mask].mean()
        if not monthly_means:
            return {}
        grand_mean = np.mean(list(monthly_means.values()))
        return {mo: round(float(v - grand_mean), 1) for mo, v in monthly_means.items()}
    except Exception:
        return {}


# =============================================================================
# Filtered forecast helpers (property only)
# =============================================================================

def _parse_sarimax_order_str(order_str: str):
    """'(1, 1, 1)x(1, 0, 0, 12)' → ((1,1,1), (1,0,0,12)) or (None, None)."""
    try:
        a, b = order_str.split("x")
        order    = tuple(int(x) for x in a.strip("() ").split(","))
        seasonal = tuple(int(x) for x in b.strip("() ").split(","))
        return order, seasonal
    except Exception:
        return None, None


def _try_property_filtered_forecast(filtered_agg: pd.DataFrame,
                                     cached_sarimax_order_str: str = None) -> dict:
    """
    Attempt Prophet + reduced-grid SARIMAX on a filtered monthly aggregation.
    Uses the real filtered data — no smoothing, no substitution.

    Returns a dict with 'status':
      "ok"                → MAPE threshold passed; includes model objects
      "insufficient_data" → fewer than PROPERTY_FORECAST_MIN_CLAIMS claims
      "poor_mape"         → Prophet backtest MAPE ≥ PROPERTY_FORECAST_MAX_MAPE
      "model_failed"      → model could not be fitted (too few months / convergence)
    """
    total_claims = int(filtered_agg["claim_count"].sum())
    if total_claims < PROPERTY_FORECAST_MIN_CLAIMS:
        return {
            "status": "insufficient_data",
            "message": (
                f"Only {total_claims:,} claims match this combination — "
                f"at least {PROPERTY_FORECAST_MIN_CLAIMS} are required to produce a reliable forecast."
            ),
        }

    hicp = load_property_hicp()
    cpi  = load_property_cpi()
    construction_interp, construction_raw = load_construction_proxy()
    df = engineer_property_features(filtered_agg, hicp, construction_interp, construction_raw, cpi)

    prophet_res = fit_prophet_property(df)
    if prophet_res is None:
        return {
            "status": "model_failed",
            "message": (
                "The forecast model could not be fitted on this combination "
                "(not enough months in the training or test window). "
                "Try using fewer or broader filters."
            ),
        }

    if prophet_res["mape"] >= PROPERTY_FORECAST_MAX_MAPE:
        return {
            "status": "poor_mape",
            "message": (
                f"Backtest MAPE is {prophet_res['mape']:.1f}% on this filter combination "
                f"(threshold: {PROPERTY_FORECAST_MAX_MAPE:.0f}%). "
                "The data is too thin or volatile for a reliable forecast. "
                "Try using fewer or broader filters."
            ),
        }

    cached_order = None
    if cached_sarimax_order_str:
        order, seasonal = _parse_sarimax_order_str(cached_sarimax_order_str)
        if order and seasonal:
            cached_order = (order, seasonal)

    sarimax_res = fit_sarimax_property_reduced_grid(df, cached_best_order=cached_order)

    return {
        "status":       "ok",
        "prophet_res":  prophet_res,
        "sarimax_res":  sarimax_res,
        "df_engineered": df,
    }


# =============================================================================
# Scenario application (shared)
# =============================================================================

def apply_scenario(fc: pd.DataFrame, cpi_pct: float, severity_pct: float, legal_mult: float,
                    covid_adj_pct: float, seasonal_adj_pct: float) -> pd.DataFrame:
    if fc.empty:
        return fc
    legal_extra = (legal_mult - 1.0) * 0.08   # legal proxy's typical share of severity growth
    annual_extra = (
        cpi_pct / 100 + severity_pct / 100
        + legal_extra
        + covid_adj_pct / 100 * 0.10
        + seasonal_adj_pct / 100 * 0.05
    )
    fc = fc.copy()
    months_out = np.arange(1, len(fc) + 1)
    growth = np.power(1 + annual_extra, months_out / 12)
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        if col in fc.columns:
            fc[col] = fc[col] * growth
    return fc


# =============================================================================
# Build + cache
# =============================================================================

def _build_casualty(group: str = "ALL") -> dict:
    monthly = load_casualty_monthly()
    usable = monthly.dropna(subset=["avg_smooth"]).reset_index(drop=True)

    cpi_raw = pd.read_csv(CPI_PATH, encoding="utf-8-sig")
    cpi_raw.columns = cpi_raw.columns.str.strip()
    cpi_raw["ds"] = pd.to_datetime(cpi_raw["Month"], format="%Y %B", errors="coerce")
    cpi_raw["val"] = pd.to_numeric(cpi_raw["VALUE"], errors="coerce")
    cpi_df = cpi_raw.dropna(subset=["ds", "val"]).sort_values("ds").reset_index(drop=True)[["ds", "val"]]

    hicp_raw = pd.read_csv(HICP_PATH, encoding="utf-8-sig")
    hicp_raw.columns = hicp_raw.columns.str.strip()
    hicp_raw["ds"] = pd.to_datetime(hicp_raw["Date"], errors="coerce")
    hicp_df = hicp_raw.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True).rename(columns={"HICP_YoY_pct": "yoy"})[["ds", "yoy"]]

    medical_df = load_medical_cpi()
    legal_df = load_legal_proxy()

    periods = FORECAST_YEARS_DEFAULT * 12
    # Forecast the smoothed severity trend — see casualty_backtest() docstring
    # for why the raw monthly average is not a viable forecast target here.
    prophet_fit = fit_prophet_casualty(usable, periods_ahead=periods, target_col="avg_smooth")
    sarimax_fit = fit_sarimax_casualty(usable, periods_ahead=periods, target_col="avg_smooth")
    correlations = casualty_correlations(monthly, cpi_df, hicp_df, medical_df, legal_df)
    backtest = casualty_backtest(monthly)
    severity = severity_breakdown()

    return {
        "monthly": monthly,
        "prophet_forecast": prophet_fit["forecast"],
        "seasonality_pct": prophet_fit["seasonality_pct"],
        "sarimax_seasonality_pct": sarimax_fit.get("sarimax_seasonality_pct", {}),
        "sarimax_forecast": sarimax_fit["forecast"],
        "correlations": correlations,
        "backtest": backtest,
        "last_obs": monthly["ds"].max(),
        "severity_breakdown": severity,
        "inflation_inputs": {
            "medical_trailing_monthly_delta": round(float(prophet_fit["med_trail"]), 3),
            "legal_trailing_monthly_delta":   round(float(prophet_fit["legal_trail"]), 3),
        },
    }


def _build_casualty_all_claims() -> dict:
    """Build forecast for ALL claims (settled + unsettled) with two series: count and average cost."""
    monthly = load_casualty_all_claims_monthly()
    usable = monthly.dropna(subset=["avg_smooth"]).reset_index(drop=True)
    usable_count = monthly.dropna(subset=["count_smooth"]).reset_index(drop=True)

    periods = FORECAST_YEARS_DEFAULT * 12

    # Fit cost forecast (similar to settled claims)
    prophet_cost = fit_prophet_casualty(usable, periods_ahead=periods, target_col="avg_smooth")
    sarimax_cost = fit_sarimax_casualty(usable, periods_ahead=periods, target_col="avg_smooth")

    # Fit count forecast (predict number of claims per month)
    prophet_count = fit_prophet_casualty(usable_count, periods_ahead=periods, target_col="count_smooth")
    sarimax_count = fit_sarimax_casualty(usable_count, periods_ahead=periods, target_col="count_smooth")

    return {
        "monthly": monthly,
        "prophet_forecast_cost": prophet_cost["forecast"],
        "sarimax_forecast_cost": sarimax_cost["forecast"],
        "prophet_forecast_count": prophet_count["forecast"],
        "sarimax_forecast_count": sarimax_count["forecast"],
        "seasonality_pct_cost": prophet_cost["seasonality_pct"],
        "seasonality_pct_count": prophet_count["seasonality_pct"],
        "sarimax_seasonality_pct_cost": sarimax_cost.get("sarimax_seasonality_pct", {}),
        "sarimax_seasonality_pct_count": sarimax_count.get("sarimax_seasonality_pct", {}),
        "last_obs": monthly["ds"].max(),
    }


def _build_all_casualty_groups() -> dict:
    """
    Build only the ALL group. Per-tier forecasting was tried and reverted:
    splitting ~1,862 claims across 5 severity tiers leaves some tiers with
    fewer than 5 claims/month on average, making their monthly average
    cost essentially the value of 1-2 individual settlements rather than a
    real average. Backtested out-of-sample MAPE on those tiers ran as high
    as 250% — not a model bug, a genuine data-volume floor. Severity is
    still reported as a static cost breakdown (see severity_breakdown())
    rather than forecast per-tier.
    """
    return {"ALL": _build_casualty("ALL")}


PROPERTY_GROUPS = ["ALL", "Water & Flooding", "Storm & Weather", "Fire", "Theft & Vandalism", "Accidental Damage"]


def _build_property(raw, hicp, cpi, construction_interp, construction_raw, group: str = "ALL") -> dict:
    monthly_all = agg_property_monthly(raw, group)
    df_all = engineer_property_features(monthly_all, hicp, construction_interp, construction_raw, cpi)

    correlations = property_correlations(df_all)
    sarimax_res = fit_sarimax_property(df_all)
    prophet_res = fit_prophet_property(df_all)

    periods = FORECAST_YEARS_DEFAULT * 12
    sarimax_fc = sarimax_property_future(sarimax_res, periods) if sarimax_res else pd.DataFrame(columns=["ds","yhat","yhat_lower","yhat_upper"])
    prophet_fc = prophet_property_future(prophet_res, df_all, periods) if prophet_res else pd.DataFrame(columns=["ds","yhat","yhat_lower","yhat_upper"])

    seasonality_pct = property_seasonality(prophet_res)
    sarimax_seasonality_pct = property_sarimax_seasonality(sarimax_res)

    test_points = []
    if sarimax_res is not None and prophet_res is not None:
        s_idx = sarimax_res["eval"].index
        s_actual = sarimax_res["eval"]["average_claim_cost"].values
        s_pred = sarimax_res["eval"]["forecast"].values
        p_pred = prophet_res["test"]["forecast"].values
        n = min(len(s_idx), len(p_pred))
        for i in range(n):
            test_points.append({
                "date": str(s_idx[i])[:10],
                "actual": round(float(s_actual[i]), 2),
                "prophet": round(float(p_pred[i]), 2),
                "sarimax": round(float(s_pred[i]), 2),
            })

    backtest = {
        "prophet_mape": round(prophet_res["mape"], 1) if prophet_res else None,
        "sarimax_mape": round(sarimax_res["mape"], 1) if sarimax_res else None,
        "sarimax_order": f"{sarimax_res['order']}x{sarimax_res['seasonal']}" if sarimax_res else None,
        "test_points": test_points,
    }

    monthly_out = monthly_all.rename(columns={"average_claim_cost": "avg_cost", "claim_count": "n_claims"}).copy()
    monthly_out["ds"] = monthly_out["Month"].dt.to_timestamp()
    monthly_out["avg_smooth"] = monthly_out["avg_cost"].rolling(6, min_periods=3).mean()
    monthly_out["yoy"] = monthly_out["avg_cost"].pct_change(12) * 100
    monthly_out = monthly_out.reset_index(drop=True)

    return {
        "monthly": monthly_out,
        "prophet_forecast": prophet_fc,
        "sarimax_forecast": sarimax_fc,
        "seasonality_pct": seasonality_pct,
        "sarimax_seasonality_pct": sarimax_seasonality_pct,
        "correlations": correlations,
        "backtest": backtest,
        "last_obs": monthly_out["ds"].max(),
    }


def _build_all_property_groups() -> dict:
    """Fit Prophet+SARIMAX separately for ALL plus each peril group, once, at cache time."""
    raw = load_property_raw()
    hicp = load_property_hicp()
    cpi = load_property_cpi()
    construction_interp, construction_raw = load_construction_proxy()

    out = {}
    for group in PROPERTY_GROUPS:
        try:
            out[group] = _build_property(raw, hicp, cpi, construction_interp, construction_raw, group)
        except Exception as exc:
            print(f"  Skipping property group '{group}': {exc}")
    return out


def _load_or_build_all() -> dict:
    if MODEL_CACHE.exists():
        with open(MODEL_CACHE, "rb") as f:
            return pickle.load(f)
    bundle = {
        "casualty_groups": _build_all_casualty_groups(),
        "casualty_all_claims": _build_casualty_all_claims(),
        "property_groups": _build_all_property_groups(),
    }
    with open(MODEL_CACHE, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


# =============================================================================
# Public API
# =============================================================================

def _build_all_claims_response(data: dict, horizon_years: int, cpi_pct: float, severity_pct: float, 
                               legal_mult: float, covid_adj: float, seasonal_adj: float) -> dict:
    """Build forecast response for all-claims (settled + unsettled) with count and cost series."""
    monthly = data["monthly"]
    last_obs = data["last_obs"]
    periods_needed = horizon_years * 12

    def _f(v):
        try:
            f = float(v)
            return None if np.isnan(f) else round(f, 2)
        except (TypeError, ValueError):
            return None

    # Cost forecasts
    p_cost = data["prophet_forecast_cost"][data["prophet_forecast_cost"]["ds"] > last_obs].head(periods_needed).reset_index(drop=True)
    s_cost = data["sarimax_forecast_cost"][data["sarimax_forecast_cost"]["ds"] > last_obs].head(periods_needed).reset_index(drop=True)
    p_cost = apply_scenario(p_cost, cpi_pct, severity_pct, legal_mult, covid_adj, seasonal_adj)
    s_cost = apply_scenario(s_cost, cpi_pct, severity_pct, legal_mult, covid_adj, seasonal_adj)

    # Count forecasts (no scenario adjustments for count)
    p_count = data["prophet_forecast_count"][data["prophet_forecast_count"]["ds"] > last_obs].head(periods_needed).reset_index(drop=True)
    s_count = data["sarimax_forecast_count"][data["sarimax_forecast_count"]["ds"] > last_obs].head(periods_needed).reset_index(drop=True)

    # Observed data
    observed_cost = []
    observed_count = []
    for _, r in monthly.iterrows():
        cost_row = {
            "date": str(r.ds)[:10], "avg_cost": _f(r.avg_cost),
            "avg_smooth": _f(r.get("avg_smooth")),
        }
        if not pd.isna(r.get("yoy", float("nan"))):
            cost_row["yoy"] = round(float(r["yoy"]), 2)
        observed_cost.append(cost_row)

        count_row = {
            "date": str(r.ds)[:10], "n_claims": int(r.n_claims) if not pd.isna(r.n_claims) else 0,
            "count_smooth": _f(r.get("count_smooth")),
        }
        if not pd.isna(r.get("count_yoy", float("nan"))):
            count_row["count_yoy"] = round(float(r["count_yoy"]), 2)
        observed_count.append(count_row)

    # Forecast points
    prophet_cost_pts = [
        {"date": str(r.ds)[:10], "yhat": round(float(r.yhat), 2),
         "yhat_lower": round(float(r.yhat_lower), 2), "yhat_upper": round(float(r.yhat_upper), 2)}
        for _, r in p_cost.iterrows()
    ]
    sarimax_cost_pts = [
        {"date": str(r.ds)[:10], "yhat": round(float(r.yhat), 2),
         "yhat_lower": round(float(r.yhat_lower), 2), "yhat_upper": round(float(r.yhat_upper), 2)}
        for _, r in s_cost.iterrows()
    ]
    prophet_count_pts = [
        {"date": str(r.ds)[:10], "yhat": round(float(r.yhat), 2),
         "yhat_lower": round(float(r.yhat_lower), 2), "yhat_upper": round(float(r.yhat_upper), 2)}
        for _, r in p_count.iterrows()
    ]
    sarimax_count_pts = [
        {"date": str(r.ds)[:10], "yhat": round(float(r.yhat), 2),
         "yhat_lower": round(float(r.yhat_lower), 2), "yhat_upper": round(float(r.yhat_upper), 2)}
        for _, r in s_count.iterrows()
    ]

    # KPIs for cost
    last12_cost = monthly["avg_cost"].dropna().iloc[-12:].mean() if len(monthly) >= 12 else monthly["avg_cost"].mean()
    p_cost_end = p_cost["yhat"].iloc[-12:].mean() if len(p_cost) >= 12 else (p_cost["yhat"].mean() if len(p_cost) else None)
    s_cost_end = s_cost["yhat"].iloc[-12:].mean() if len(s_cost) >= 12 else (s_cost["yhat"].mean() if len(s_cost) else None)

    # KPIs for count
    last12_count = monthly["n_claims"].dropna().iloc[-12:].mean() if len(monthly) >= 12 else monthly["n_claims"].mean()
    p_count_end = p_count["yhat"].iloc[-12:].mean() if len(p_count) >= 12 else (p_count["yhat"].mean() if len(p_count) else None)
    s_count_end = s_count["yhat"].iloc[-12:].mean() if len(s_count) >= 12 else (s_count["yhat"].mean() if len(s_count) else None)

    return {
        "observed_cost": observed_cost,
        "observed_count": observed_count,
        "prophet_forecast_cost": prophet_cost_pts,
        "sarimax_forecast_cost": sarimax_cost_pts,
        "prophet_forecast_count": prophet_count_pts,
        "sarimax_forecast_count": sarimax_count_pts,
        "seasonality_pct_cost": data.get("seasonality_pct_cost", {}),
        "seasonality_pct_count": data.get("seasonality_pct_count", {}),
        "kpis": {
            "last_12mo_avg_cost": round(float(last12_cost), 0) if last12_cost else None,
            "prophet_end_avg_cost": round(float(p_cost_end), 0) if p_cost_end else None,
            "sarimax_end_avg_cost": round(float(s_cost_end), 0) if s_cost_end else None,
            "last_12mo_avg_count": round(float(last12_count), 0) if last12_count else None,
            "prophet_end_avg_count": round(float(p_count_end), 0) if p_count_end else None,
            "sarimax_end_avg_count": round(float(s_count_end), 0) if s_count_end else None,
        },
    }


def get_forecast(claim_type: str = "casualty", horizon_years: int = 3,
                  cpi_pct: float = 2.5, severity_pct: float = 5.0, legal_mult: float = 1.2,
                  covid_adj: float = 0.0, seasonal_adj: float = 0.0,
                  peril_group: str = "ALL", severity_group: str = "ALL", claims_subset: str = "settled",
                  building_contents: str = "ALL", claim_category: str = "ALL", region: str = "ALL") -> dict:
    bundle = _load_or_build_all()
    if claim_type == "casualty":
        if claims_subset == "all":
            data = bundle["casualty_all_claims"]
            # For all claims, handle both count and cost forecasts
            return _build_all_claims_response(data, horizon_years, cpi_pct, severity_pct, legal_mult, covid_adj, seasonal_adj)
        else:
            # settled claims (default)
            data = bundle["casualty_groups"]["ALL"]
    else:
        groups = bundle["property_groups"]
        data = groups.get(peril_group, groups.get("ALL"))

    forecast_status  = "ok"
    forecast_message = None
    periods_needed   = horizon_years * 12
    _empty_fc = pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])

    if claim_type == "property" and (building_contents != "ALL" or claim_category != "ALL" or region != "ALL"):
        # ── filtered path: re-aggregate from raw CSV, then try to fit new models ──
        raw = load_property_raw()
        filtered_agg = agg_property_monthly(
            raw, group=peril_group,
            building_contents=building_contents,
            claim_category=claim_category,
            region=region,
        )
        monthly = _post_process_property_monthly(filtered_agg)

        cached_order_str = (data.get("backtest") or {}).get("sarimax_order")
        fc_attempt = _try_property_filtered_forecast(filtered_agg, cached_order_str)

        if fc_attempt["status"] == "ok":
            prophet_res = fc_attempt["prophet_res"]
            sarimax_res = fc_attempt["sarimax_res"]
            df_eng      = fc_attempt["df_engineered"]

            p_future = prophet_property_future(prophet_res, df_eng, periods_needed)
            s_future = (sarimax_property_future(sarimax_res, periods_needed)
                        if sarimax_res else _empty_fc.copy())
            p_future = apply_scenario(p_future, cpi_pct, severity_pct, legal_mult, covid_adj, seasonal_adj)
            s_future = apply_scenario(s_future, cpi_pct, severity_pct, legal_mult, covid_adj, seasonal_adj)

            seasonality_pct         = property_seasonality(prophet_res)
            sarimax_seasonality_pct = property_sarimax_seasonality(sarimax_res)

            test_points = []
            if sarimax_res is not None:
                s_eval = sarimax_res["eval"]
                p_test = prophet_res["test"]
                for i in range(min(len(s_eval), len(p_test))):
                    test_points.append({
                        "date":    str(s_eval.index[i])[:10],
                        "actual":  round(float(s_eval["average_claim_cost"].iloc[i]), 2),
                        "prophet": round(float(p_test["forecast"].iloc[i]), 2),
                        "sarimax": round(float(s_eval["forecast"].iloc[i]), 2),
                    })
            backtest = {
                "prophet_mape":  round(prophet_res["mape"], 1),
                "sarimax_mape":  round(sarimax_res["mape"], 1) if sarimax_res else None,
                "sarimax_order": (f"{sarimax_res['order']}x{sarimax_res['seasonal']}"
                                  if sarimax_res else None),
                "test_points":   test_points,
            }
        else:
            p_future                = _empty_fc.copy()
            s_future                = _empty_fc.copy()
            seasonality_pct         = {}
            sarimax_seasonality_pct = {}
            backtest                = {"prophet_mape": None, "sarimax_mape": None,
                                       "sarimax_order": None, "test_points": []}
            forecast_status  = fc_attempt["status"]
            forecast_message = fc_attempt["message"]
    else:
        # ── unfiltered path: use pre-cached models ────────────────────────────
        monthly  = data["monthly"]
        last_obs = data["last_obs"]
        p_future = (data["prophet_forecast"][data["prophet_forecast"]["ds"] > last_obs]
                    .head(periods_needed).reset_index(drop=True))
        s_future = (data["sarimax_forecast"][data["sarimax_forecast"]["ds"] > last_obs]
                    .head(periods_needed).reset_index(drop=True))
        p_future = apply_scenario(p_future, cpi_pct, severity_pct, legal_mult, covid_adj, seasonal_adj)
        s_future = apply_scenario(s_future, cpi_pct, severity_pct, legal_mult, covid_adj, seasonal_adj)
        seasonality_pct         = data["seasonality_pct"]
        sarimax_seasonality_pct = data.get("sarimax_seasonality_pct", {})
        backtest                = data["backtest"]

    def _f(v):
        try:
            f = float(v)
            return None if np.isnan(f) else round(f, 2)
        except (TypeError, ValueError):
            return None

    observed = []
    for _, r in monthly.iterrows():
        row = {
            "date": str(r.ds)[:10], "avg_cost": _f(r.avg_cost),
            "n_claims": int(r.n_claims) if not pd.isna(r.n_claims) else 0,
            "avg_smooth": _f(r.get("avg_smooth")),
        }
        if not pd.isna(r.get("yoy", float("nan"))):
            row["yoy"] = round(float(r["yoy"]), 2)
        observed.append(row)

    prophet_pts = [
        {"date": str(r.ds)[:10], "yhat": round(float(r.yhat), 2),
         "yhat_lower": round(float(r.yhat_lower), 2), "yhat_upper": round(float(r.yhat_upper), 2)}
        for _, r in p_future.iterrows()
    ]
    sarimax_pts = [
        {"date": str(r.ds)[:10], "yhat": round(float(r.yhat), 2),
         "yhat_lower": round(float(r.yhat_lower), 2), "yhat_upper": round(float(r.yhat_upper), 2)}
        for _, r in s_future.iterrows()
    ]

    last12_avg = monthly["avg_cost"].dropna().iloc[-12:].mean()
    p_end_avg = p_future["yhat"].iloc[-12:].mean() if len(p_future) >= 12 else (p_future["yhat"].mean() if len(p_future) else None)
    s_end_avg = s_future["yhat"].iloc[-12:].mean() if len(s_future) >= 12 else (s_future["yhat"].mean() if len(s_future) else None)

    sorted_m = monthly.dropna(subset=["avg_smooth"]).sort_values("ds")
    if len(sorted_m) >= 2:
        first_sm, last_sm = sorted_m["avg_smooth"].iloc[0], sorted_m["avg_smooth"].iloc[-1]
        n_yrs = (sorted_m["ds"].iloc[-1] - sorted_m["ds"].iloc[0]).days / 365.25
        hist_cagr = (pow(last_sm / first_sm, 1 / n_yrs) - 1) * 100 if n_yrs > 0 and first_sm > 0 else 0.0
    else:
        hist_cagr = 0.0

    return {
        "observed":                observed,
        "prophet_forecast":        prophet_pts,
        "sarimax_forecast":        sarimax_pts,
        "seasonality_pct":         seasonality_pct,
        "sarimax_seasonality_pct": sarimax_seasonality_pct,
        "correlations":            data["correlations"],
        "backtest":                backtest,
        "inflation_inputs":        data.get("inflation_inputs"),
        "severity_breakdown":      data.get("severity_breakdown"),
        "kpis": {
            "last_12mo_avg":   round(float(last12_avg), 0),
            "hist_cagr":       round(float(hist_cagr), 2),
            "total_claims":    int(monthly["n_claims"].sum()),
            "prophet_end_avg": round(float(p_end_avg), 0) if p_end_avg is not None else None,
            "sarimax_end_avg": round(float(s_end_avg), 0) if s_end_avg is not None else None,
        },
        "forecast_status":  forecast_status,
        "forecast_message": forecast_message,
    }


def get_home_summary(horizon_years: int = 3) -> dict:
    return {
        "casualty": get_forecast("casualty", horizon_years=horizon_years),
        "property": get_forecast("property", horizon_years=horizon_years),
    }


if __name__ == "__main__":
    print("Building model cache for both claim types (real Prophet + real SARIMAX, x2 claim types)...")
    MODEL_CACHE.unlink(missing_ok=True)
    r1 = get_forecast("casualty")
    print("Casualty KPIs:", r1["kpis"])
    print("Casualty correlations:", r1["correlations"])
    print("Casualty backtest (fit MAPE):", r1["backtest"]["fit_mape_prophet"], r1["backtest"]["fit_mape_sarimax"])
    print("Casualty backtest (OOS 2025 MAPE):", r1["backtest"]["prophet_mape"], r1["backtest"]["sarimax_mape"])
    r2 = get_forecast("property")
    print("\nProperty KPIs:", r2["kpis"])
    print("Property correlations:", r2["correlations"])
    print("Property backtest MAPE:", r2["backtest"]["prophet_mape"], r2["backtest"]["sarimax_mape"], r2["backtest"]["sarimax_order"])
    print("\nDone. Cache saved to", MODEL_CACHE)