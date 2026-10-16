# Haze Tracker SG: Technical Write-up

---

## 1. Overview

**Haze Tracker SG** is a live Singapore haze dashboard with an hourly forecast. It displays current PM2.5 and PSI readings for the five NEA regions, provides a cigarette-equivalent exposure estimate, and forecasts the island-wide mean 1-hour PSI at **+1, +2, +3, +6, and +24 hours**, each accompanied by an **80% prediction interval**.

The project is split into two repositories:
* **`Haze-Tracker-Model-Training`**: An offline Python pipeline (data backfill, feature engineering, training, and evaluation) that exports a single `model.json` file.
* **`Haze-Tracker`**: A static single-page site (`index.html`, favicon assets) combined with serverless endpoints in `functions/` (`/api/air`, `/api/forecast`) hosted on Cloudflare Pages.

### The Core Concept
Haze in Singapore is primarily a transport problem. Smoke from fires in Sumatra and Kalimantan arrives hours to days later depending on wind patterns, and is eventually cleared by rain and atmospheric mixing. Consequently, the model predicts the **change from the current reading** rather than predicting PM2.5 from scratch, leveraging fire detections, wind-based arrival estimates, weather forecasts, and upstream air quality.

---

## 2. Architecture

```text
backfill.py  ->  data/*.csv  ->  features.py  ->  trainnew.py  ->  out/model.json
(NEA, ERA5, FIRMS, CAMS)        (53 features)     (2 regimes x 5 horizons)
                                                                      |
                                          Haze-Tracker/functions  <----+  (model evaluated in JS)  [confirm]
                                                    |
                                  /api/air + /api/forecast  ->  index.html
```

The design choice that holds this together is that **`features.py` serves as the single definition of model inputs**. `model.json` ships a spec block containing point coordinates, bearings, decay constants, window definitions, feature lists, and plain-language descriptions of each formula. The serving code reproduces these features exactly, ensuring training and serving paths never silently drift apart.

---

## 3. Data Pipeline (`backfill.py`)

All data is downloaded into `./data`, cached under `data/raw/`, and the script is fully resumable: if interrupted by a rate limit or a Wi-Fi drop, rerunning it fetches only missing data.

### Sources
* **NEA PM2.5 and 24-hour PSI**: Sourced from `data.gov.sg`, hourly per region (North, South, East, West, Central) starting from April 2014. Because the historical dataset lags by weeks, a `pm25tail` step bridges the gap using the real-time API, checking overlaps and warning if timestamps look offset.
* **Weather**: Sourced from the Open-Meteo archive (ERA5) at seven points: Singapore plus six smoke-source regions (Riau, Jambi, South Sumatra, West Kalimantan, Central Kalimantan, and South Kalimantan). Variables include temperature, humidity, precipitation, wind speed/direction, and boundary layer height (BLH). If the archive rejects BLH with an HTTP 400 error, the script falls back to an older variable set and trains without it.
* **Fire Hotspots**: Sourced from NASA FIRMS (MODIS Terra and Aqua) over a bounding box covering Sumatra, Kalimantan, and Peninsular Malaysia ($95^\circ\text{E}$ to $119^\circ\text{E}$, $6.5^\circ\text{S}$ to $6^\circ\text{N}$). The API serves at most 10 days per call, switching to the near-real-time product beyond 75 days.
* **Upstream PM2.5**: Sourced from the Open-Meteo air-quality API (CAMS reanalysis) at six points between and around the sources and Singapore: Batam, Bintan, Dumai, Pekanbaru, Jambi, and Pontianak.

### Design Decisions & Rationale
* **MODIS Only**: MODIS is the sole sensor set consistent from 2014 to the present. Higher-resolution sensors exhibit varying detection sensitivities before and after their launch dates, which would introduce artificial shifts in fire activity metrics.
* **Single Weather Provider**: Training and serving use the same weather provider to eliminate unit or bias mismatches.
* **Robust HTTP Layer**: Features retries with exponential backoff, honors `Retry-After` headers, and applies distinct handling for `429` and `5xx` errors versus other `4xx` errors.
* **Sanity Report**: The `summary` function prints hours, missing percentages, maximum values, and hours exceeding 55 and 150 $\mu\text{g/m}^3$ per year, verifying at a glance that major haze years (2015, 2019, 2023, 2026) are correctly captured prior to training.

---

## 4. Feature Engineering (`features.py`)

Up to **53 features** are computed from trailing windows that include the origin hour. Windows are strict: if any hour in the window is missing, the feature evaluates to `NaN`, and training uses complete rows only. Concentrations and counts are log1p-transformed throughout.

* **Target and Aggregation**: The target series uses the mean of the five regions (`AGG = "mean"`, with `"max"` available for a worst-region variant). An hour is valid only if at least 3 of the 5 regions reported.

### Feature Groups
1. **Persistence and History**: Current, 3-hour, 24-hour, and 72-hour PM2.5, alongside the 6-hour change.
2. **Local Weather at Singapore**: 6-hour mean wind components, speed, humidity, temperature, and 6- and 24-hour rainfall.
3. **Source-Region Dryness**: 7-day and 30-day rainfall and 24-hour humidity averaged across source points (dry peat results in more smoke production).
4. **Fire Push**: 24-hour hotspot counts per source cell, weighted by distance decay ($\exp(-\text{km}/800)$) and the positive part of the 12-hour mean wind component along the source-to-Singapore bearing. Includes lagged versions at 24 and 48 hours, as well as fire-weighted components aligned along and across the wind vector.
5. **Forecast Weather (NWP)**: Mean wind, projected fire push, rainfall, and BLH over the next 3 and 6 hours.
6. **Seasonality**: Sine and cosine transformations of the day of the year and hour of the day.
7. **Excess Interactions**: `pm_excess` measures the amount by which current levels exceed $40\ \mu\text{g/m}^3$ (in log space) and interacts with fire push, wind, rain, forecast push, and forecast BLH, enabling linear models to adapt when haze is already present.
8. **Arrival-Time Features (v5)**: For each fire detection, the code estimates smoke transit time to Singapore and aggregates contributions into time bins.
9. **Upstream Station Features**: Advected push from each upstream PM2.5 station ($\text{decay} \times \text{station PM2.5} \times \text{positive wind towards Singapore}$), plus concentration gradients (station minus Singapore) for Batam, Dumai, and Pekanbaru.

### How Arrival-Time Features Work
For every fire detection:
* Computes the inverse-distance-weighted trailing 6-hour wind across the seven points.
* Projects the wind onto the bearing from the detection to Singapore (along and cross components).
* Estimates travel time $\eta = \text{distance} / (3.6 \times \text{along})$, discarding detections with an `along` wind below $0.5\text{ m/s}$ or an $\eta$ exceeding 72 hours.
* Weights by Fire Radiative Power (default 20, capped at 500), distance decay, an angular term $\exp(-(\theta / 30^\circ)^2)$ accounting for angular wind deviation, and a rain washout term $\exp(-\text{rain}/2)$ using local 3-hour rainfall.
* Contributions are added to arrival-time bins relative to forecast origin: already arrived in the past 6 hours, and arriving in 0–1, 1–3, and 3–6 hours. This is implemented efficiently using a difference-array trick (`np.add.at` followed by `cumsum`) rather than an hour-by-hour loop.

### Forecast-Error Simulation (`noisy()`)
Training uses ERA5 reanalysis for future-weather features, but live serving relies on real forecasts, which are less accurate. To prevent the model from over-trusting future-weather inputs, `noisy()` degrades reanalysis winds, rain, and BLH using autocorrelated noise (AR(1) with coefficient $0.9$ to simulate persistent hourly forecast errors), log-normal rain error, and log-normal BLH error. A scaling multiplier (e.g., `python trainnew.py data 1.0`) writes outputs to dedicated folders (`out/`, `out_noise0.5/`, etc.) for sensitivity comparisons.

---

## 5. Model Training (`trainnew.py`)

* **Formulation**: For each horizon $h \in \{1, 2, 3, 6, 24\}$, the target is the log change $\log(1 + \text{PM}[t+h]) - \log(1 + \text{PM}[t])$. The baseline represents persistence ("same as now"), and the model learns only the residual correction. Predictions are converted back using `expm1`.
* **Two Regimes**: Separate models are fitted for **clean** and **haze** conditions. An hour is classified as haze if the current mean exceeds $35\ \mu\text{g/m}^3$ or the 24-hour mean exceeds $25\ \mu\text{g/m}^3$, separating the distinct physical dynamics of low-signal clean air versus fire-driven haze.
* **Candidates per Regime and Horizon**:
  * `ridge_full` and `ridge_small`: Standardized Ridge regression ($\alpha \in \{10, 100, 1000\}$).
  * `xgb_full` and `xgb_small`: Conservative XGBoost regressors (depth 2 or 3, learning rate $0.02$, 80 trees, subsample $0.7$, column sample $0.7$, L1 $0.1$, L2 $10$, min child weight $5$).
  * The small set consists of 19 hand-picked features (core persistence, fire push, current wind/rain, and forecast-driven features). The full set includes all 53 features.
* **Sample Weighting**: XGBoost samples with current PM2.5 above $35\ \mu\text{g/m}^3$ receive a weight of $2.5$ to prevent under-prediction of extreme peaks.
* **Validation**: Leave-one-month-out cross-validation with a 7-day purge on both sides of every test month to prevent temporal data leakage from autocorrelated neighboring hours. Hyperparameters are selected via out-of-fold RMSE in log space, and the winning candidate per regime is chosen by out-of-fold MAE in $\mu\text{g/m}^3$.
* **Uncertainty**: The 80% interval uses out-of-fold residual 99-quantiles per regime, defining the interval as the point forecast plus the 10th and 90th percentile residuals, dynamically widening or narrowing with regime and horizon.
* **Exceedance Probability**: The probability that PM2.5 exceeds $55\ \mu\text{g/m}^3$ (NEA Band I/II boundary) is derived from residual quantiles and recalibrated via isotonic regression into a 41-point lookup table.
* **Export**: `out/model.json` (schema version 6) bundles the feature list, per-horizon/per-regime models (Ridge weights, XGBoost JSON tree dumps), residual quantiles, calibration tables, performance metrics, and model spec.

---

## 6. Evaluation Results

> [!NOTE]
> *Placeholder for pipeline benchmark outputs: MAE versus persistence per horizon, above-55 recall, PSI-band accuracy, and noise-multiplier comparisons.*

---

## 7. The Website (`Haze-Tracker`)

The front-end consists of a single static `index.html` (~775 lines) with no framework or build step, utilizing vanilla JavaScript, inline SVG charts, and CSS variables for PSI band colors. It loads in a single request plus a Google Font for optimal performance on mobile devices.

### Key Sections
* **Hero**: Worst-region (or user-selected region) PSI with 1-hour, 3-hour, or 24-hour toggles, featuring a dynamic SVG avatar reflecting the current band.
* **By Region**: Raw PM2.5 alongside 1-hour and 3-hour PSI (calculated client-side from NEA PM2.5 breakpoints) and official NEA 24-hour PSI.
* **Cigarette Equivalent**: Berkeley Earth's rule-of-thumb metric ($22\ \mu\text{g/m}^3$ over 24 hours per cigarette) with a region selector and outdoor-time slider (indoor air assumed at 40% of outdoor concentration), persisted in `localStorage`.
* **Haze Forecast**: Horizon-by-horizon PSI, 80% range, band tables, a 24-hour history plus forecast chart, a "biggest factor" explanation, and model metadata.
* **Past 24 Hours**: Regional charts with hover tooltips for PM2.5 or PSI across 1-, 3-, or 24-hour averaging windows.
* **What to Do**: Health guidance following NEA logic (1-hour PM2.5 for immediate actions, 24-hour PSI for long-term planning), segmented for healthy adults, sensitive groups, and chronic conditions.
* **Data Sources**: Live health monitoring panel tracking status feeds (green, amber, red).

### Technical Highlights
* **Client-Side PSI Calculation**: Matches training breakpoints ($0, 12, 55, 150, 250, 350, 500\ \mu\text{g/m}^3$ mapped to $0, 50, 100, 200, 300, 400, 500$) for exact alignment across charts, tables, and forecasts.
* **Cache-First Rendering**: Cached `localStorage` states render immediately with shimmer skeletons while fresh data loads in the background.
* **Polite Refresh Scheduler**: Fetches data only if the current hour is missing, then sleeps until 3 minutes past the hour. Implements exponential retries ($2, 3, 5, 10, 15\text{ minutes}$) and pauses when browser tabs are hidden.
* **Explainable Forecasts**: Estimates input group impacts by perturbing features against typical values, converting them into PSI changes using local breakpoint slopes.
* **Privacy-Preserving Geolocation**: Runs entirely client-side using a Singapore bounding box mapped to 55 named towns, discarding coordinate logs.

---

## 8. Progression (Reconstructed)

1. **Baseline**: NEA PM2.5 history, local weather, and basic trailing-window features evaluated against persistence.
2. **Fire Transport**: Distance- and wind-weighted hotspot counts projected onto source-to-Singapore bearings (`fire_push`) with lagged features.
3. **Forecast Weather**: Future wind/rain features with noise simulation to prevent reanalysis over-trust; BLH added as an optional fallback input.
4. **Regimes & Model Families**: Separation into clean and haze regimes; XGBoost introduced alongside Ridge with haze-weighted sample boosting.
5. **v5 Arrival Times**: Per-detection arrival bins extended with rain washout terms.
6. **Upstream Stations**: CAMS PM2.5 at six upstream nodes with advected push and gradient features.
7. **Model Schema v6**: Current `model.json` implementation with regime-specific quantiles and isotonic calibration.
8. **Web Interface**: Developed across 41 commits covering dashboards, forecasts, health panels, geolocation, and SEO metadata.

---

## 9. Limitations & Cleanup Notes

* **Reanalysis vs. Live Data**: Training on ERA5/CAMS reanalysis with synthetic noise approximates, but does not perfectly replicate, real NWP forecasts and live CAMS analyses.
* **Haze Event Scarcity**: The haze regime relies on a limited number of episodes (2015, 2019, 2023, 2026), leading to higher variance for extreme predictions.
* **In-Sample Selection**: Hyperparameter tuning and model selection share out-of-fold validation splits, which can introduce slight optimism bias.
* **Selection Guards**: In `trainnew.py`, selection condition checks include `or True`, disabling `MIN_GAIN` and `RECALL_TOL` guardrails.
* **Island-Wide Target**: Forecasts reflect island-wide means, which may diverge from individual regional readings.

---

## 10. Reproducing the Build

```bash
pip install -r requirements.txt
export FIRMS_MAP_KEY=your_nasa_firms_key
python backfill.py              # Or selective: python backfill.py pm25 weather upstream hotspots summary
python trainnew.py data 1.0     # Args: data folder, forecast-error multiplier
# -> Generates out/model.json, copy into the site's functions/ folder and deploy
```

---

## 11. File Map

* `backfill.py`: Downloads and caches NEA, ERA5, FIRMS, and CAMS datasets.
* `features.py`: Single source of truth for model inputs and embedded model specs.
* `trainnew.py`: Handles regime splits, candidate search, leave-one-month-out cross-validation, uncertainty/calibration, and `model.json` export.
* `compare.py`: Model comparison utilities.
* `index.html`: Complete front-end interface.
* `functions/`: Serverless endpoints (`/api/air`, `/api/forecast`).