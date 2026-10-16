#!/usr/bin/env python3
"""
backfill.py - step 1: download all raw training data into ./data

    pip install -r requirements.txt
    export FIRMS_MAP_KEY=your_nasa_firms_key      (Windows: set FIRMS_MAP_KEY=...)
    python backfill.py                  # everything
    python backfill.py pm25 psi         # just some parts: pm25 psi pm25tail weather hotspots summary

Resumable: every download is cached under data/raw/, so if it stops (rate limits,
Wi-Fi) just run it again and it only fetches what's missing.

Outputs (all timestamps are Singapore time unless noted):
    data/pm25_hourly.csv     ts, north, south, east, west, central      (NEA, from Apr 2014)
    data/psi24_hourly.csv    same layout, official 24h PSI              (NEA, from Apr 2014)
    data/weather_hourly.csv  ts, point, temp, rh, precip, wspd, wdir    (Open-Meteo/ERA5, SG + 6 source regions)
    data/hotspots.csv        raw MODIS hotspots over Sumatra/Kalimantan (FIRMS; acq_date/acq_time are UTC)
"""
import os, sys, time, json
import datetime as dt
from pathlib import Path
import requests
import pandas as pd

DATA = Path("data"); RAW = DATA / "raw"
for p in (DATA, RAW / "weather", RAW / "firms"):
    p.mkdir(parents=True, exist_ok=True)

DATASETS = {  # data.gov.sg dataset ids
    "pm25":  "d_6c0d5fc34838b12472475fdb73c0af29",  # Historical 1-hr PM2.5   (Apr 2014 ->)
    "psi24": "d_b4cf557f8750260d229c49fd768e11ed",  # Historical 24-hr PSI    (Apr 2014 ->)
}
REGIONS = ["north", "south", "east", "west", "central"]

# Weather is pulled for Singapore plus 6 smoke-source points, all from Open-Meteo, so the
# model is trained and served on the same provider.
POINTS = {
    "sg":     (1.35,  103.82),
    "riau":   (0.50,  101.45),
    "jambi":  (-1.60, 103.60),
    "sumsel": (-3.00, 104.75),
    "kalbar": (-0.03, 109.33),
    "kalteng":(-2.21, 113.92),
    "kalsel": (-3.32, 114.59),
}
HOURLY_VARS = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m"
RENAME = {"time": "ts", "temperature_2m": "temp", "relative_humidity_2m": "rh",
          "precipitation": "precip", "wind_speed_10m": "wspd", "wind_direction_10m": "wdir", "boundary_layer_height": "blh"}

START = dt.date(2014, 1, 1)          # PM2.5 begins Apr 2014; the extra months give lag buffer
BBOX = (95.0, -6.5, 119.0, 6.0)      # west, south, east, north: Sumatra, Kalimantan, Peninsular Malaysia


# ---------------------------------------------------------------- http
def http_get(url, params=None, tries=7, timeout=90):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as e:
            wait = min(2 ** i * 2, 120); print(f"  network error {e!r}; retrying in {wait}s"); time.sleep(wait); continue
        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get("Retry-After", min(2 ** i * 5, 300)))
            print(f"  HTTP {r.status_code} from {url.split('?')[0]}; waiting {wait:.0f}s"); time.sleep(wait); continue
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} from {r.url}\n{r.text[:300]}")
        return r
    raise RuntimeError(f"giving up on {url}")


# ---------------------------------------------------------------- data.gov.sg
def fetch_datastore(name, rid):
    out = RAW / f"{name}_raw.csv"
    if out.exists():
        print(f"{name}: using cached {out}"); return pd.read_csv(out)
    rows, offset, total = [], 0, None
    while True:
        j = http_get("https://data.gov.sg/api/action/datastore_search",
                     {"resource_id": rid, "limit": 5000, "offset": offset}).json()
        res = j["result"]; recs = res["records"]; total = res.get("total", total)
        if not recs: break
        rows.extend(recs); offset += len(recs)       # advance by what we actually got
        print(f"  {name}: {offset}/{total}")
        if total is not None and offset >= total: break
        time.sleep(0.5)
    df = pd.DataFrame(rows).drop(columns=["_id"], errors="ignore")
    df.to_csv(out, index=False)
    return df


def tidy_regional(df, name):
    """Turn whatever layout data.gov.sg gives us into: ts + one column per region."""
    cols = {c.lower(): c for c in df.columns}
    tcol = next((cols[k] for k in ("timestamp", "datetime", "date", "time", "ts") if k in cols), None) \
        or next((c for lc, c in cols.items() if any(k in lc for k in ("date", "time"))), None)
    reg = {k: c for lc, c in cols.items() for k in REGIONS if k in lc}
    if tcol is None:
        sys.exit(f"[{name}] couldn't find a timestamp column. Columns are: {list(df.columns)}")
    if len(reg) < 5 and "region" in cols:            # long layout: one row per (time, region)
        vcol = next(c for c in df.columns if c not in (tcol, cols["region"]))
        df = df.pivot_table(index=tcol, columns=cols["region"], values=vcol, aggfunc="mean").reset_index()
        cols = {c.lower(): c for c in df.columns}
        reg = {k: c for lc, c in cols.items() for k in REGIONS if k in lc}
    if len(reg) < 5:
        sys.exit(f"[{name}] expected 5 region columns, found {list(reg)}. Columns are: {list(df.columns)}\n"
                 f"Edit REGIONS / tidy_regional() to match.")
    s = df[tcol].astype(str)
    ts = pd.to_datetime(s, dayfirst="/" in s.iloc[0], errors="coerce")
    try: ts = ts.dt.tz_localize(None)                # drop offset, keep local wall time
    except (TypeError, AttributeError): pass
    out = pd.DataFrame({"ts": ts})
    for k in REGIONS: out[k] = pd.to_numeric(df[reg[k]], errors="coerce")
    out = out.dropna(subset=["ts"]).drop_duplicates("ts").sort_values("ts")
    return out


def get_regional(key):
    raw = fetch_datastore(key, DATASETS[key])
    print(f"  columns: {list(raw.columns)}")
    out = tidy_regional(raw, key)
    path = DATA / ("pm25_hourly.csv" if key == "pm25" else "psi24_hourly.csv")
    out.to_csv(path, index=False)
    print(f"{key}: {len(out):,} rows, {out.ts.min()} -> {out.ts.max()}  ->  {path}")


# ---------------------------------------------------------------- PM2.5 tail
def fill_pm25_tail():
    """The historical dataset lags by weeks; the real-time API (same one your site uses) fills the gap."""
    path = DATA / "pm25_hourly.csv"
    pm = pd.read_csv(path, parse_dates=["ts"])
    d, today, rows = pm.ts.max().date(), dt.date.today(), []
    print(f"pm25tail: history ends {pm.ts.max()}; fetching {d} -> {today} from the real-time API")
    while d <= today:
        token = None
        for _ in range(10):
            params = {"date": d.isoformat()}
            if token: params["paginationToken"] = token
            j = http_get("https://api-open.data.gov.sg/v2/real-time/api/pm25", params).json()
            for it in j["data"]["items"]:
                rd = it["readings"]["pm25_one_hourly"]
                rows.append({"ts": it["timestamp"], **{k: rd.get(k) for k in REGIONS}})
            token = j["data"].get("paginationToken")
            if not token: break
        d += dt.timedelta(days=1); time.sleep(0.7)
    new = pd.DataFrame(rows)
    new["ts"] = pd.to_datetime(new["ts"], errors="coerce")
    try: new["ts"] = new["ts"].dt.tz_localize(None)
    except (TypeError, AttributeError): pass
    new = new.dropna(subset=["ts"]).drop_duplicates("ts")
    # timestamp-convention check: where the two sources overlap, values should match
    both = pm.merge(new, on="ts", suffixes=("", "_new"))
    if len(both):
        same = (both[REGIONS].values == both[[r + "_new" for r in REGIONS]].values).mean()
        print(f"pm25tail: {len(both)} overlapping hours, {same:.0%} identical values "
              f"({'timestamps line up' if same > 0.9 else 'WARNING: timestamps may be offset, check before training'})")
    out = pd.concat([pm, new]).drop_duplicates("ts", keep="first").sort_values("ts")
    for k in REGIONS: out[k] = pd.to_numeric(out[k], errors="coerce")
    out.to_csv(path, index=False)
    print(f"pm25tail: now {len(out):,} rows, {out.ts.min()} -> {out.ts.max()}")


# ---------------------------------------------------------------- weather
BLH_VAR = "boundary_layer_height"

def fetch_weather():
    """Open-Meteo archive (ERA5). Tries to include boundary layer height; if the archive refuses that variable
    (HTTP 400) it falls back to the older variable set and tells you, so nothing breaks."""
    out = DATA / "weather_hourly.csv"
    end = dt.date.today() - dt.timedelta(days=7)      # the archive lags ~5 days
    frames, use_blh = [], True
    def request(lat, lon, s, e, variables):
        return http_get("https://archive-api.open-meteo.com/v1/archive", {
            "latitude": lat, "longitude": lon, "start_date": s.isoformat(), "end_date": e.isoformat(),
            "hourly": variables, "timezone": "Asia/Singapore", "wind_speed_unit": "ms"})
    for name, (lat, lon) in POINTS.items():
        for y in range(START.year, end.year + 1):
            s, e = max(dt.date(y, 1, 1), START), min(dt.date(y, 12, 31), end)
            f_blh, f_old = RAW / "weather" / f"{name}_{y}_blh.json", RAW / "weather" / f"{name}_{y}.json"
            if use_blh and not f_blh.exists():
                try:
                    print(f"weather: {name} {y} (with boundary layer height)")
                    f_blh.write_text(request(lat, lon, s, e, HOURLY_VARS + "," + BLH_VAR).text); time.sleep(3)
                except RuntimeError as err:
                    if "HTTP 400" not in str(err): raise
                    print(f"  the archive does not offer {BLH_VAR} ({str(err)[:120]}); continuing without it"); use_blh = False
            if use_blh: f = f_blh
            else:
                f = f_old
                if not f.exists():
                    print(f"weather: {name} {y}"); f.write_text(request(lat, lon, s, e, HOURLY_VARS).text); time.sleep(3)
            d = pd.DataFrame(json.loads(f.read_text())["hourly"]).rename(columns=RENAME)
            d["point"] = name; frames.append(d)
    w = pd.concat(frames, ignore_index=True)
    w["ts"] = pd.to_datetime(w["ts"])
    w.to_csv(out, index=False)
    has = "blh" in w.columns and w["blh"].notna().any()
    print(f"weather: {len(w):,} rows, {w.ts.min()} -> {w.ts.max()}  ->  {out}   |  boundary layer height: {'YES' if has else 'NO (model will train without it)'}")


# ---------------------------------------------------------------- hotspots
def fetch_hotspots():
    key = os.environ.get("FIRMS_MAP_KEY") or sys.exit("Set the FIRMS_MAP_KEY environment variable first.")
    today = dt.date.today(); d = START; frames = []
    area = ",".join(str(x) for x in BBOX)
    # MODIS (Terra+Aqua) is used because it is the one consistent sensor set across 2014-now.
    # Recent days are only in the near-real-time product; older ones in standard processing.
    while d <= today:
        n = min(10, (today - d).days + 1)             # the API serves at most 10 days per call
        src = "MODIS_NRT" if (today - d).days < 75 else "MODIS_SP"
        f = RAW / "firms" / f"{src}_{d}_{n}.csv"
        if not f.exists():
            txt = http_get(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/{area}/{n}/{d}").text
            if txt.strip() and not txt.lstrip().startswith("latitude"):
                sys.exit(f"Unexpected FIRMS reply for {src} {d}: {txt[:200]}")
            f.write_text(txt); time.sleep(0.3)
            if d.day <= 10: print(f"hotspots: {d}")
        if f.stat().st_size > 0:
            frames.append(pd.read_csv(f))
        d += dt.timedelta(days=n)
    h = pd.concat(frames, ignore_index=True)
    keep = [c for c in ("latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence", "frp", "daynight") if c in h]
    h = h[keep].drop_duplicates()
    h.to_csv(DATA / "hotspots.csv", index=False)
    print(f"hotspots: {len(h):,} detections, {h.acq_date.min()} -> {h.acq_date.max()}  ->  data/hotspots.csv")


# ---------------------------------------------------------------- sanity report
def summary():
    pm = pd.read_csv(DATA / "pm25_hourly.csv", parse_dates=["ts"]).set_index("ts")
    worst = pm[REGIONS].max(axis=1)
    print("\nPM2.5 by year (check the haze years are there: 2015, 2019, 2023, 2026)")
    print(f"{'year':<6}{'hours':>8}{'missing%':>10}{'max':>7}{'hrs>55':>8}{'hrs>150':>9}")
    for y, g in worst.groupby(worst.index.year):
        full = 366 * 24 if y % 4 == 0 else 365 * 24
        print(f"{y:<6}{g.notna().sum():>8}{100 * (1 - g.notna().sum() / full):>10.1f}{g.max():>7.0f}{(g > 55).sum():>8}{(g > 150).sum():>9}")
    print("\nWorst hour on record:", worst.idxmax(), "->", worst.max(), "ug/m3")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["pm25", "psi", "weather", "hotspots", "summary"]
    if "pm25" in steps: get_regional("pm25")
    if "psi" in steps or "psi24" in steps: get_regional("psi24")
    if "pm25tail" in steps: fill_pm25_tail()
    if "weather" in steps: fetch_weather()
    if "hotspots" in steps: fetch_hotspots()
    if "summary" in steps: summary()

# --- Upstream Regional Stations (South, Southwest, and Southeast of Singapore) ---
UPSTREAM_STATIONS = {
    "batam": (1.1301, 104.0529),          # Batam, Riau Islands (~30 km SE)[cite: 10]
    "bintan": (0.9167, 104.4500),         # Tanjung Pinang, Bintan (~75 km SE)
    "dumai": (1.6680, 101.4450),          # Coastal Riau, Sumatra (~260 km W)
    "pekanbaru": (0.5333, 101.4500),      # Inland Riau, Sumatra (~280 km WSW)
    "jambi": (-1.6100, 103.6100),         # Jambi, Sumatra (~330 km SSW)
    "pontianak": (-0.0300, 109.3300),     # Pontianak, West Kalimantan (~630 km ESE)
}


# ---------------------------------------------------------------- upstream station PM2.5
# ---------------------------------------------------------------- upstream station PM2.5
def fetch_upstream():
    """
    Fetch historical hourly PM2.5 from Open-Meteo Air Quality CAMS reanalysis
    for nearby stations south, southwest, and southeast of Singapore.
    """
    out = DATA / "upstream_pm25_hourly.csv"
    up_dir = RAW / "upstream"
    up_dir.mkdir(parents=True, exist_ok=True)
    end = dt.date.today()
    frames = []

    for name, (lat, lon) in UPSTREAM_STATIONS.items():
        for y in range(START.year, end.year + 1):
            s, e = max(dt.date(y, 1, 1), START), min(dt.date(y, 12, 31), end)
            f = up_dir / f"{name}_{y}.json"

            # Fetch if file doesn't exist or is empty (0 bytes)
            if not f.exists() or f.stat().st_size == 0:
                print(f"upstream: {name} {y}")
                r = http_get("https://air-quality-api.open-meteo.com/v1/air-quality", {
                    "latitude": lat, "longitude": lon, "start_date": s.isoformat(), "end_date": e.isoformat(),
                    "hourly": "pm2_5", "timezone": "Asia/Singapore"})
                f.write_text(r.text, encoding="utf-8")
                time.sleep(1)

            # Safely load JSON; re-fetch if corrupted
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                print(f"upstream: corrupted cache found for {name} {y}, re-fetching...")
                f.unlink(missing_ok=True)
                r = http_get("https://air-quality-api.open-meteo.com/v1/air-quality", {
                    "latitude": lat, "longitude": lon, "start_date": s.isoformat(), "end_date": e.isoformat(),
                    "hourly": "pm2_5", "timezone": "Asia/Singapore"})
                f.write_text(r.text, encoding="utf-8")
                data = json.loads(f.read_text(encoding="utf-8"))

            if "hourly" in data and "pm2_5" in data["hourly"]:
                d = pd.DataFrame(data["hourly"]).rename(columns={"time": "ts", "pm2_5": name})
                frames.append(d)

    dfs = {}
    for name in UPSTREAM_STATIONS:
        st_frames = [f for f in frames if name in f.columns]
        if st_frames:
            df = pd.concat(st_frames, ignore_index=True)
            df["ts"] = pd.to_datetime(df["ts"])
            dfs[name] = df.set_index("ts")[name]

    if dfs:
        merged = pd.DataFrame(dfs).sort_index().reset_index()
        merged.to_csv(out, index=False)
        print(f"upstream: {len(merged):,} rows, {merged.ts.min()} -> {merged.ts.max()}  ->  {out}")


# In __main__ section of backfill.py:
if __name__ == "__main__":
    steps = sys.argv[1:] or [
        "pm25",
        "psi",
        "weather",
        "upstream",
        "hotspots",
        "summary",
    ]  #[cite: 10]
    # ...
    if "upstream" in steps:
        fetch_upstream()