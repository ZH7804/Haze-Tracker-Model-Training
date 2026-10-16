#!/usr/bin/env python3
"""
compare.py - diagnostic: does ANY model beat "same as now" once haze is already present?
Run:  python compare.py        (reads ./data, writes out/compare.txt; takes a few minutes)

Every candidate predicts the change in log1p(PM2.5) from now, scored leave-one-year-out (one-week gap)
on hours that are already in the haze regime (PM2.5 above 40 now, or 24h average above 30).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import features as ft

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = Path("out"); OUT.mkdir(exist_ok=True)
HORIZONS = [1, 3, 6]; PURGE_H = 7 * 24
LOG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

F, pm, pm_mean, wx, counts, _, _d = ft.build(DATA)
FEATS = [c for c in ft.FEATURES if F[c].notna().any()]; F = F[FEATS]
SMALL = [c for c in ["pm_now", "pm_3h", "pm_24h", "pm_chg6", "fire_push", "fire_push_l24", "sg_wspd", "sg_rain6", "pm_excess", "wind_along_fw", "wind_cross_fw", "fut_push3", "fut_push6", "fut_u3", "fut_v3", "fut_rain3", "fut_blh3", "fut_blh6", "ex_x_fpush6"] + ft.ARR_FEATURES if c in FEATS]
hs_of = lambda X: ((X["pm_now"] > np.log1p(40)) | (X["pm_24h"] > np.log1p(30))).values

def ridge(cols, alpha, haze_only):
    def run(Xtr, dtr, Xte):
        if haze_only: m = hs_of(Xtr); Xtr, dtr = Xtr[m], dtr[m]
        w = np.where(hs_of(Xtr), 5.0, 1.0)
        p = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Xtr[cols], dtr, ridge__sample_weight=w)
        return p.predict(Xte[cols])
    return run

def gbt(haze_only):
    def run(Xtr, dtr, Xte):
        if haze_only: m = hs_of(Xtr); Xtr, dtr = Xtr[m], dtr[m]
        w = np.where(hs_of(Xtr), 5.0, 1.0)
        g = HistGradientBoostingRegressor(loss="absolute_error", max_depth=3, learning_rate=0.05, max_iter=200,
                                          min_samples_leaf=50, l2_regularization=1.0, random_state=0)
        return g.fit(Xtr, dtr, sample_weight=w).predict(Xte)
    return run

CANDS = {
    "persistence": lambda Xtr, dtr, Xte: np.zeros(len(Xte)),
    "persist_3h_avg": lambda Xtr, dtr, Xte: (Xte["pm_3h"] - Xte["pm_now"]).values,   # "same as the last 3h average"
    "ridge_all (v2)": ridge(FEATS, 1000, False),
    "ridge_haze": ridge(FEATS, 100, True),
    "ridge_haze_small": ridge(SMALL, 100, True),
    "gbt_haze": gbt(True),
    "gbt_all": gbt(False),
}

for h in HORIZONS:
    y = np.log1p(pm_mean.shift(-h)); ok = F.notna().all(axis=1) & y.notna()
    X = F[ok]; d = y[ok] - X["pm_now"]; hs = hs_of(X); yrs = X.index.year
    oof = {k: pd.Series(np.nan, index=X.index) for k in CANDS}
    for yr in sorted(set(yrs)):
        te = yrs == yr
        lo, hi = X.index[te].min() - pd.Timedelta(hours=PURGE_H), X.index[te].max() + pd.Timedelta(hours=PURGE_H)
        tr = ~te & ~((X.index >= lo) & (X.index <= hi))
        tes = te & hs
        if tes.sum() < 30 or tr.sum() < 5000: continue
        for k, fn in CANDS.items():
            oof[k][tes] = fn(X[tr], d[tr], X[tes])
    valid = oof["persistence"].notna().values
    base = X["pm_now"].values
    true = np.expm1(base + d.values)
    hz = true > 55
    say(f"\n=========== +{h}h, scored on {valid.sum():,} hours already in the haze regime ({(valid & hz).sum()} with PM2.5 > 55) ===========")
    say(f"{'candidate':<18}{'MAE haze-regime':>16}{'MAE >55':>10}{'recall>55':>11}{'precision':>11}   2015 / 2019 / 2026 MAE")
    res = {}
    for k in CANDS:
        pred = np.expm1(base + oof[k].fillna(0).values)
        m = valid; a = np.abs(pred - true)
        rec = ((pred > 55) & hz & m).sum() / max((hz & m).sum(), 1); prec = ((pred > 55) & hz & m).sum() / max(((pred > 55) & m).sum(), 1)
        yr = lambda y_: a[m & (yrs == y_)].mean() if (m & (yrs == y_)).any() else float("nan")
        res[k] = a[m].mean()
        say(f"{k:<18}{a[m].mean():>16.2f}{a[m & hz].mean():>10.2f}{rec:>11.2f}{prec:>11.2f}   {yr(2015):.1f} / {yr(2019):.1f} / {yr(2026):.1f}")
    bl = min(("persistence", "persist_3h_avg"), key=res.get)            # strongest simple baseline
    best = min((k for k in res if k not in ("persistence", "persist_3h_avg")), key=res.get)
    gain = 100 * (1 - res[best] / res[bl])
    say(f"strongest simple baseline: {bl}. best model: {best} ({gain:+.1f}% MAE vs that baseline in the haze regime) -> "
        f"{'worth keeping' if gain > 3 else 'no meaningful gain: a simple baseline is as good as anything we have'}")
(OUT / "compare.txt").write_text("\n".join(LOG))
