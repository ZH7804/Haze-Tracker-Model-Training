"""
features.py - the ONE definition of the model's inputs.

Calculates inputs using trailing windows, forecast NWP weather, fire hotspot arrival estimates,
and wind-direction-projected upstream regional station advection.
"""
import numpy as np, pandas as pd
from pathlib import Path
from backfill import POINTS, REGIONS

SG = POINTS["sg"]
SRC = [p for p in POINTS if p != "sg"]
HS_LATENCY_H = 3
HS_MIN_CONF = 30
DECAY_KM = 800.0
WIND_WIN = 12
AGG = "mean"          # "mean" of the 5 regions, or "max" (worst region)
EXCESS_PM = 40
FUTURE_H = 6          # how many hours of forecast weather the model may look at

# --- Upstream Regional Stations (South, Southwest, and Southeast of Singapore) ---
UPSTREAM_STATIONS = {
    "batam":     (1.1301,  104.0529),     # Batam (~30 km SE)
    "bintan":    (0.9167,  104.4500),     # Bintan (~75 km SE)
    "dumai":     (1.6680,  101.4450),     # Coastal Riau (~260 km W)
    "pekanbaru": (0.5333,  101.4500),     # Inland Riau (~280 km WSW)
    "jambi":     (-1.6100, 103.6100),     # Jambi (~330 km SSW)
    "pontianak": (-0.0300, 109.3300),     # West Kalimantan (~630 km ESE)
}
UP_SRC = list(UPSTREAM_STATIONS.keys())

# --- v5: per-hotspot arrival-time features ---
ARR = {"wind_avg_h": 6, "idw_eps_km2": 1.0, "along_min": 0.5, "theta0_deg": 30.0,
       "eta_max_h": 72.0, "frp_default": 20.0, "frp_cap": 500.0}
ARR_BINS = {"arr_past6": (-6, 0), "arr_0_1": (0, 1), "arr_1_3": (1, 3), "arr_3_6": (3, 6)}
ARR_FEATURES = ["arr_past6", "arr_0_1", "arr_1_3", "arr_3_6", "arr_next6", "ex_x_arr6"]

UPSTREAM_FEATURES = [
    "up_push",            # Total wind-advected PM2.5 push from all upstream stations
    "up_push_batam",      # Wind-advected PM2.5 push specifically from Batam
    "up_grad_batam",      # Concentration gradient: log1p(Batam) - log1p(Singapore_now)
    "up_grad_dumai",      # Concentration gradient: log1p(Dumai) - log1p(Singapore_now)
    "up_grad_pekanbaru", # Concentration gradient: log1p(Pekanbaru) - log1p(Singapore_now)
]

FEATURES = [
    "pm_now", "pm_3h", "pm_24h", "pm_72h", "pm_chg6",
    "sg_u", "sg_v", "sg_wspd", "sg_rh", "sg_temp", "sg_rain6", "sg_rain24",
    "src_rain_7d", "src_rain_30d", "src_rh_24h",
    "fire_tot24", "fire_push", "fire_push_l24", "fire_push_l48",
    "doy_sin", "doy_cos", "hod_sin", "hod_cos",
    "pm_excess", "ex_x_fire", "ex_x_wind", "ex_x_rain",
    "wind_along_fw", "wind_cross_fw",
    "fut_u3", "fut_v3", "fut_u6", "fut_v6",
    "fut_push3", "fut_push6",
    "fut_rain3", "fut_rain6",
    "blh_now", "fut_blh3", "fut_blh6",
    "ex_x_fpush6", "ex_x_fblh6",
    "arr_past6", "arr_0_1", "arr_1_3", "arr_3_6", "arr_next6", "ex_x_arr6",
    # --- Upstream Regional Station Features ---
    *UPSTREAM_FEATURES
]
BLH_FEATURES = ["blh_now", "fut_blh3", "fut_blh6", "ex_x_fblh6"]


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


BEAR = {p: float(bearing_deg(*POINTS[p], *SG)) for p in SRC}
DIST = {p: float(haversine_km(*POINTS[p], *SG)) for p in SRC}
DECAY = {p: float(np.exp(-DIST[p] / DECAY_KM)) for p in SRC}

UP_BEAR = {p: float(bearing_deg(*UPSTREAM_STATIONS[p], *SG)) for p in UP_SRC}
UP_DIST = {p: float(haversine_km(*UPSTREAM_STATIONS[p], *SG)) for p in UP_SRC}
UP_DECAY = {p: float(np.exp(-UP_DIST[p] / DECAY_KM)) for p in UP_SRC}


def spec(has_blh=False, noise=1.0):
    return {"points": {k: list(v) for k, v in POINTS.items()}, "source_points": SRC, "regions": REGIONS,
            "upstream_stations": {k: list(v) for k, v in UPSTREAM_STATIONS.items()},
            "upstream_bearing": UP_BEAR, "upstream_distance_km": UP_DIST, "upstream_decay": UP_DECAY,
            "bearing_to_sg": BEAR, "distance_km": DIST, "decay": DECAY, "decay_km": DECAY_KM,
            "aggregation": AGG, "excess_pm": EXCESS_PM, "hotspot_latency_h": HS_LATENCY_H, "hotspot_min_confidence": HS_MIN_CONF,
            "hotspot_bbox_wsen": [95.0, -6.5, 119.0, 6.0], "wind_window_h": WIND_WIN, "future_hours": FUTURE_H,
            "has_blh": bool(has_blh), "training_forecast_noise": noise,
            "hotspot_cell": "each detection is assigned to the nearest source point (haversine); counts binned by SGT hour of (detection + latency)",
            "wind_components": "u=-wspd*sin(dir), v=-wspd*cos(dir) (dir = direction wind comes FROM); along = u*sin(b)+v*cos(b), cross = u*cos(b)-v*sin(b), b = bearing source->SG",
            "windows": "trailing, inclusive of origin hour, strict (any missing hour -> NaN); fut_* use hours t+1..t+n",
            "arrival_bins": {k: list(v) for k, v in ARR_BINS.items()},
            "arrival": ARR,
            "arrival_doc": "per detection: wind=IDW of trailing-6h mean u,v at the 7 points; along/cross vs bearing detection->SG; eta_h=dist_km/(3.6*along); contrib=frp*exp(-dist/decay_km)*exp(-(theta/theta0)^2); bin if eta - (t - t_det) in (lo,hi]",
            "features": FEATURES}


def load_inputs(data="data"):
    d = Path(data)
    pm = pd.read_csv(d / "pm25_hourly.csv", parse_dates=["ts"])
    pm["ts"] = pm["ts"].dt.floor("h")
    pm = pm.drop_duplicates("ts").set_index("ts").sort_index()
    idx = pd.date_range(pm.index.min(), pm.index.max(), freq="h")
    pm = pm.reindex(idx)
    ok = pm[REGIONS].notna().sum(axis=1) >= 3
    pm_mean = (pm[REGIONS].max(axis=1) if AGG == "max" else pm[REGIONS].mean(axis=1)).where(ok)

    w = pd.read_csv(d / "weather_hourly.csv", parse_dates=["ts"])
    vars_ = ["temp", "rh", "precip", "wspd", "wdir"] + (["blh"] if "blh" in w.columns else [])
    wx = {v: w.pivot(index="ts", columns="point", values=v).reindex(idx) for v in vars_}

    h = pd.read_csv(d / "hotspots.csv")
    conf = pd.to_numeric(h["confidence"], errors="coerce")
    h = h[conf.isna() | (conf >= HS_MIN_CONF)].copy()
    at = h["acq_time"].astype(int)
    t = pd.to_datetime(h["acq_date"]) + pd.to_timedelta(at // 100, unit="h") + pd.to_timedelta(at % 100, unit="m")
    h["hr"] = (t + pd.Timedelta(hours=8 + HS_LATENCY_H)).dt.floor("h")
    dist = np.stack([haversine_km(h["latitude"].values, h["longitude"].values, *POINTS[p]) for p in SRC])
    h["cell"] = np.array(SRC)[dist.argmin(axis=0)]
    counts = h.groupby(["hr", "cell"]).size().unstack(fill_value=0).reindex(idx, fill_value=0).reindex(columns=SRC, fill_value=0)
    frp = pd.to_numeric(h["frp"], errors="coerce") if "frp" in h else pd.Series(np.nan, index=h.index)
    dets = pd.DataFrame({"hr": h["hr"].values, "lat": h["latitude"].values, "lon": h["longitude"].values, "frp": frp.values})

    # Load backfilled upstream PM2.5 CSV if present
    up_file = d / "upstream_pm25_hourly.csv"
    if up_file.exists():
        up_df = pd.read_csv(up_file, parse_dates=["ts"])
        up_df["ts"] = up_df["ts"].dt.floor("h")
        up_df = up_df.drop_duplicates("ts").set_index("ts").sort_index()
        upstream_pm = up_df.reindex(idx)
    else:
        upstream_pm = None

    return pm, pm_mean, wx, counts, dets, upstream_pm


def noisy(wx, s, seed=0):
    if s == 0: return wx
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed); out = {k: v.copy() for k, v in wx.items()}
    n = len(next(iter(wx.values())))
    ar = lambda sd: lfilter([1.0], [1.0, -0.9], rng.normal(0, sd * np.sqrt(1 - 0.81), n))
    for p in POINTS:
        th = np.deg2rad(wx["wdir"][p].values); sp = wx["wspd"][p].values
        u = -sp * np.sin(th) + ar(1.2 * s); v = -sp * np.cos(th) + ar(1.2 * s)
        out["wspd"][p] = np.hypot(u, v); out["wdir"][p] = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
        out["precip"][p] = wx["precip"][p].values * np.exp(rng.normal(0, 0.7 * s, n))
        if "blh" in wx: out["blh"][p] = wx["blh"][p].values * np.exp(ar(0.25 * s))
    return out


def compute(pm_mean, wx, counts, wx_f, dets, upstream_pm=None):
    r = lambda s, n: s.rolling(n, min_periods=n)
    fw = lambda s, n, how: (r(s, n).mean() if how == "mean" else r(s, n).sum()).shift(-n)
    idx = pm_mean.index
    F = pd.DataFrame(index=idx)
    pm3 = r(pm_mean, 3).mean()
    F["pm_now"] = np.log1p(pm_mean)
    F["pm_3h"] = np.log1p(pm3)
    F["pm_24h"] = np.log1p(r(pm_mean, 24).mean())
    F["pm_72h"] = np.log1p(r(pm_mean, 72).mean())
    F["pm_chg6"] = F["pm_3h"] - F["pm_3h"].shift(6)

    uv = lambda W, p: (-W["wspd"][p] * np.sin(np.deg2rad(W["wdir"][p])), -W["wspd"][p] * np.cos(np.deg2rad(W["wdir"][p])))
    u, v = uv(wx, "sg")
    F["sg_u"], F["sg_v"] = r(u, 6).mean(), r(v, 6).mean()
    F["sg_wspd"] = r(wx["wspd"]["sg"], 6).mean()
    F["sg_rh"] = r(wx["rh"]["sg"], 6).mean()
    F["sg_temp"] = r(wx["temp"]["sg"], 6).mean()
    F["sg_rain6"] = np.log1p(r(wx["precip"]["sg"], 6).sum())
    F["sg_rain24"] = np.log1p(r(wx["precip"]["sg"], 24).sum())

    P = wx["precip"][SRC]
    F["src_rain_7d"] = np.log1p(r(P, 168).sum().mean(axis=1))
    F["src_rain_30d"] = np.log1p(r(P, 720).sum().mean(axis=1))
    F["src_rh_24h"] = r(wx["rh"][SRC], 24).mean().mean(axis=1)

    fires24 = r(counts, 24).sum()
    push, num_a, num_c, den = 0, 0, 0, 0
    proj_f = {}
    for p in SRC:
        b = np.deg2rad(BEAR[p])
        up, vp = uv(wx, p)
        along, cross = up * np.sin(b) + vp * np.cos(b), up * np.cos(b) - vp * np.sin(b)
        al12, cr12 = r(along, WIND_WIN).mean(), r(cross, WIND_WIN).mean()
        wgt = DECAY[p] * fires24[p]
        push = push + wgt * al12.clip(lower=0)
        num_a, num_c, den = num_a + wgt * al12, num_c + wgt * cr12, den + wgt
        uf, vf = uv(wx_f, p); proj_f[p] = uf * np.sin(b) + vf * np.cos(b)
    F["fire_tot24"] = np.log1p(fires24.sum(axis=1, skipna=False))
    F["fire_push"] = np.log1p(push)
    F["fire_push_l24"] = F["fire_push"].shift(24)
    F["fire_push_l48"] = F["fire_push"].shift(48)
    F["wind_along_fw"] = (num_a / den).where(den > 0, 0.0).where(den.notna())
    F["wind_cross_fw"] = (num_c / den).where(den > 0, 0.0).where(den.notna())

    uf, vf = uv(wx_f, "sg")
    for n in (3, 6):
        F[f"fut_u{n}"], F[f"fut_v{n}"] = fw(uf, n, "mean"), fw(vf, n, "mean")
        F[f"fut_push{n}"] = np.log1p(sum(DECAY[p] * fires24[p] * fw(proj_f[p], n, "mean").clip(lower=0) for p in SRC))
        F[f"fut_rain{n}"] = np.log1p(fw(wx_f["precip"]["sg"], n, "sum"))
    if "blh" in wx:
        F["blh_now"] = np.log1p(wx["blh"]["sg"])
        F["fut_blh3"], F["fut_blh6"] = np.log1p(fw(wx_f["blh"]["sg"], 3, "mean")), np.log1p(fw(wx_f["blh"]["sg"], 6, "mean"))
    else:
        for c in ("blh_now", "fut_blh3", "fut_blh6"): F[c] = np.nan

    doy = idx.dayofyear.values; hod = idx.hour.values
    F["doy_sin"], F["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    F["hod_sin"], F["hod_cos"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)

    ex = (F["pm_now"] - np.log1p(EXCESS_PM)).clip(lower=0)
    F["pm_excess"] = ex
    F["ex_x_fire"], F["ex_x_wind"], F["ex_x_rain"] = ex * F["fire_push"], ex * F["sg_wspd"], ex * F["sg_rain6"]
    F["ex_x_fpush6"], F["ex_x_fblh6"] = ex * F["fut_push6"], ex * F["fut_blh6"]

    A = arrival_sums(dets, wx, idx)
    for k in ARR_BINS: F[k] = np.log1p(pd.Series(A[k], index=idx))
    nxt = A["arr_0_1"] + A["arr_1_3"] + A["arr_3_6"]
    F["arr_next6"] = np.log1p(pd.Series(nxt, index=idx))
    F["ex_x_arr6"] = ex * F["arr_next6"]

    # --- Upstream Regional Station Features (Wind Direction Projected) ---
    if upstream_pm is not None and not upstream_pm.empty:
        up_push_total = 0
        for p in UP_SRC:
            b = np.deg2rad(UP_BEAR[p])
            # Projection of Singapore local wind vector along station -> Singapore bearing line
            along_p = r(u * np.sin(b) + v * np.cos(b), WIND_WIN).mean()
            pm_p = upstream_pm[p] if p in upstream_pm else pd.Series(np.nan, index=idx)
            
            # Advection push = decay * station_PM25 * positive_wind_toward_SG
            push_p = UP_DECAY[p] * pm_p.clip(lower=0) * along_p.clip(lower=0)
            up_push_total = up_push_total + push_p

            if p == "batam":
                F["up_push_batam"] = np.log1p(push_p)
                F["up_grad_batam"] = np.log1p(pm_p) - F["pm_now"]
            elif p == "dumai":
                F["up_grad_dumai"] = np.log1p(pm_p) - F["pm_now"]
            elif p == "pekanbaru":
                F["up_grad_pekanbaru"] = np.log1p(pm_p) - F["pm_now"]

        F["up_push"] = np.log1p(up_push_total)
    else:
        for c in UPSTREAM_FEATURES: F[c] = np.nan

    return F[FEATURES]


def build(data="data", noise=1.0):
    pm, pm_mean, wx, counts, dets, upstream_pm = load_inputs(data)
    wx_f = noisy(wx, noise)
    return compute(pm_mean, wx, counts, wx_f, dets, upstream_pm), pm, pm_mean, wx, counts, wx_f, dets

# In featuresnew.py, update the arrival_sums function:

def arrival_sums(dets, wx, idx):
    n = len(idx); out = {k: np.zeros(n + 1) for k in ARR_BINS}
    pos = idx.get_indexer(pd.DatetimeIndex(dets["hr"]))
    keep = pos >= 0
    pos = pos[keep]; lat = dets["lat"].values[keep]; lon = dets["lon"].values[keep]; frp = dets["frp"].values[keep]
    names = list(POINTS)
    cols = []
    rain_cols = [] # NEW: track precipitation for washout
    
    for p in names:
        u, v = -wx["wspd"][p] * np.sin(np.deg2rad(wx["wdir"][p])), -wx["wspd"][p] * np.cos(np.deg2rad(wx["wdir"][p]))
        cols.append((u.rolling(ARR["wind_avg_h"], min_periods=ARR["wind_avg_h"]).mean().values,
                     v.rolling(ARR["wind_avg_h"], min_periods=ARR["wind_avg_h"]).mean().values))
        # NEW: Get rolling 3h rain at the source to determine washout
        rain_cols.append(wx["precip"][p].rolling(3, min_periods=1).sum().values)
        
    U = np.column_stack([c[0] for c in cols])[pos]; V = np.column_stack([c[1] for c in cols])[pos]
    R = np.column_stack([r for r in rain_cols])[pos] # NEW: Rain matrix
    
    d_pt = np.stack([haversine_km(lat, lon, *POINTS[p]) for p in names], axis=1)
    w = 1.0 / (d_pt ** 2 + ARR["idw_eps_km2"])
    
    u = (w * U).sum(1) / w.sum(1); v = (w * V).sum(1) / w.sum(1)
    local_rain = (w * R).sum(1) / w.sum(1) # NEW: IDW of source rain
    
    dist = haversine_km(lat, lon, *SG); b = np.deg2rad(bearing_deg(lat, lon, *SG))
    along = u * np.sin(b) + v * np.cos(b); cross = u * np.cos(b) - v * np.sin(b)
    
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.degrees(np.arctan2(np.abs(cross), along))
        eta = dist / (3.6 * along)
        ok = np.isfinite(along) & (along > ARR["along_min"]) & (eta <= ARR["eta_max_h"])
        
    f = np.clip(np.where(np.isnan(frp), ARR["frp_default"], frp), 0, ARR["frp_cap"])
    
    # NEW: Apply rain washout penalty (e.g., exp(-local_rain/2)). 
    # If it rained 5mm, smoke is severely reduced before traveling.
    rain_penalty = np.exp(-local_rain / 2.0) 
    
    c = f * np.exp(-dist / DECAY_KM) * np.exp(-(theta / ARR["theta0_deg"]) ** 2) * rain_penalty
    
    a = pos + eta
    for k, (lo, hi) in ARR_BINS.items():
        s0 = np.maximum(np.ceil(a[ok] - hi), pos[ok]); e0 = np.ceil(a[ok] - lo)
        m = e0 > s0
        s0 = np.clip(s0[m], 0, n).astype(int); e0 = np.clip(e0[m], 0, n).astype(int); cm = c[ok][m]
        np.add.at(out[k], s0, cm); np.add.at(out[k], e0, -cm)
    return {k: np.maximum(np.cumsum(v)[:n], 0.0) for k, v in out.items()}