# Claims Cost Inflation Index

A forecasting web app that models how property and casualty insurance claim costs will move over time, built on a real Irish P&C insurer's claims dataset. Combines two independent forecasting models (Prophet and SARIMAX) with macroeconomic regressors - CPI, HICP, medical cost inflation, and a legal cost proxy - to project claim severity 6-36 months ahead under different economic scenarios.

Built end-to-end as a consultancy deliverable during my MSc at Trinity College Dublin.

## What it does

- **Forecasts claim cost inflation** for two lines of business - Property (buildings + contents, by peril) and Casualty (settled claims) - using both Prophet and SARIMAX models in parallel, so the two can be compared against each other.
- **Lets users stress-test scenarios** - adjust CPI assumptions, severity uplift, legal cost multipliers, and COVID sensitivity via sliders, or apply preset scenarios (Low inflation / Central / High pressure).
- **Backtests model accuracy** - both models are trained on data through 2024 and evaluated against actual 2025 outcomes (true out-of-sample), with MAPE reported for both.
- **Surfaces the "why"** - correlation analysis between claim severity and inflation indices, seasonality decomposition, and a severity-tier cost breakdown, all rendered as interactive Chart.js visualizations.

## Tech stack

- **Backend:** FastAPI (Python), serving a forecasting API and three HTML pages
- **Forecasting:** [Prophet](https://facebook.github.io/prophet/) (Meta) and SARIMAX (`statsmodels`), both fit on a log scale
- **Data processing:** pandas, numpy, scipy
- **Frontend:** Vanilla JS + Chart.js (no framework - kept deliberately lightweight)

## Methodology highlights

- Both models are fit on **log-transformed** claim cost, with results back-transformed for display - this keeps forecasts non-negative and handles the right-skew typical of claims severity data.
- **Casualty** uses a COVID-period exogenous regressor (Mar 2020 – Jun 2021) plus medical CPI and a legal-cost proxy (built from CSO professional services earnings, since Ireland has no published legal services price index).
- **Property** uses HICP and a construction-sector wage proxy as regressors, with future regressor values projected forward using trailing 12-month slope (since true future inflation isn't observable at forecast time).
- Per-severity-tier forecasting was tested and **deliberately reverted** - several injury tiers had too few monthly claims to forecast reliably (out-of-sample MAPE exceeded 200% on thin tiers). Severity is instead reported as a static cost breakdown rather than forecast individually - a real example of stopping at the point where added model complexity stops adding signal.
- Model accuracy is validated with a genuine **2025 holdout backtest** (trained on 2020–2024 only), not just in-sample fit.

## Project structure

```
├── app.py              # FastAPI routes + forecast API
├── model.py            # Data loading, feature engineering, Prophet/SARIMAX fitting
├── static/
│   ├── charts.js        # Chart.js builders (forecast, seasonality, correlation, backtest)
│   └── style.css
├── templates/
│   ├── home.html         # Combined summary view
│   ├── property.html     # Property deep-dive
│   └── casualty.html     # Casualty deep-dive
└── requirements.txt
```

## Running locally

```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Then visit `http://localhost:8000`.

> Note: the original claims dataset is excluded from this repo for confidentiality. The code is structured to run against any CSV with the same schema (settlement/incident dates, incurred reserve amounts, claim categories).

## Limitations / honest caveats

- Forecast confidence intervals widen substantially toward the end of the 36-month horizon - Prophet's CI in particular should be read directionally, not as a precise bound.
- The legal cost proxy is an approximation (Ireland has no official legal services price index) - flagged as such in the app's model guide rather than presented as ground truth.
- Property's construction cost regressor is also a proxy (average weekly earnings in construction, rebased to 100), used in place of a dedicated construction materials cost index that wasn't available.
