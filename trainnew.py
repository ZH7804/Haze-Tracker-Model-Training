#!/usr/bin/env python3
"""
train.py - two-regime model for +1h, +3h, +6h supporting Ridge and XGBoost.
"""
from xgboost import XGBRegressor
import json, sys, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import features as ft

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
NOISE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0     
OUT = Path("out" if NOISE == 1.0 else f"out_noise{NOISE:g}"); OUT.mkdir(exist_ok=True)
HORIZONS = [1,2,3,6,24]
ALPHAS = [10, 100, 1000]
PURGE_H = 7 * 24
MIN_GAIN = 0.02
RECALL_TOL = 0.03      
LOG_EXCESS, LOG_24 = np.log1p(35), np.log1p(25)
SMALL_BASE = ["pm_now", "pm_3h", "pm_24h", "pm_chg6", "fire_push", "fire_push_l24", "sg_wspd", "sg_rain6", "pm_excess"]   
SMALL_NWP = SMALL_BASE + ["wind_along_fw", "wind_cross_fw", "fut_push3", "fut_push6", "fut_u3", "fut_v3", "fut_rain3", "fut_blh3", "fut_blh6", "ex_x_fpush6"]# + ft.UPSTREAM_FEATURES + ft.ARR_FEATURES

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

F, pm, pm_mean, wx, counts, wx_f, dets = ft.build(DATA, NOISE)
FEATS = [c for c in ft.FEATURES if F[c].notna().any()]
HAS_BLH = "blh_now" in FEATS
F = F[FEATS]

CANDS = {
    "ridge_full": {"cols": FEATS, "model": "ridge"},
    "ridge_small": {"cols": [c for c in SMALL_NWP if c in FEATS], "model": "ridge"},
    "xgb_full": {"cols": FEATS, "model": "xgb"},
    "xgb_small": {"cols": [c for c in SMALL_NWP if c in FEATS], "model": "xgb"}
}

say(f"predicting: {ft.AGG} of the 5 regions | forecast-error multiplier {NOISE:g} | boundary layer height: {'yes' if HAS_BLH else 'no'}")
say(f"feature rows: {len(F):,} ({F.index.min()} -> {F.index.max()}) | {len(FEATS)} features")

BP_C = [0, 12, 55, 150, 250, 350, 500]; BP_P = [0, 50, 100, 200, 300, 400, 500]
band_of = lambda c: np.searchsorted([50, 100, 200, 300], np.round(np.interp(c, BP_C, BP_P)))
hs_of = lambda X: ((X["pm_now"] > LOG_EXCESS) | (X["pm_24h"] > LOG_24)).values
mae = lambda a, b, m=slice(None): float(np.abs(a[m] - b[m]).mean())

def fit_model(Xtr, dtr, cols, model_type, param):
    if model_type == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=param)).fit(Xtr[cols], dtr)
    else:
        # Weight haze hours higher so XGBoost doesn't under-predict extreme peaks
        sample_weights = np.where(Xtr["pm_now"] > np.log1p(35), 2.5, 1.0)
        
        model = XGBRegressor(
            max_depth=int(param),
            learning_rate=0.02,
            n_estimators=80,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=10.0,
            min_child_weight=5,
            n_jobs=-1,
            random_state=42
        )
        return model.fit(Xtr[cols], dtr, sample_weight=sample_weights)

def loyo(X, d, mask, cols, model_type, param):
    """Leave-one-month-out predictions using Year-Month tuples as folds"""
    oof = pd.Series(np.nan, index=X.index)
    ym_tuples = [(idx.year, idx.month) for idx in X.index]
    unique_ym = sorted(list(set(ym_tuples)))
    
    for yr, mo in unique_ym:
        in_fold = (X.index.year == yr) & (X.index.month == mo)
        fold_start = X.index[in_fold].min() - pd.Timedelta(hours=PURGE_H)
        fold_end = X.index[in_fold].max() + pd.Timedelta(hours=PURGE_H)
        
        # Purge 7 days before and after the test month to prevent temporal leakage
        tr = mask & ~in_fold & ~((X.index >= fold_start) & (X.index <= fold_end))
        te = mask & in_fold
        
        if te.sum() < 10 or tr.sum() < 100: 
            continue
            
        oof[te] = fit_model(X[tr], d[tr], cols, model_type, param).predict(X[te][cols])
    return oof

def f1(pred, true):
    tp = (pred & true).sum(); p = tp / max(pred.sum(), 1); r = tp / max(true.sum(), 1)
    return p, r

models, metrics, summary = {}, {}, []
for h in HORIZONS:
    y = np.log1p(pm_mean.shift(-h)); ok = F.notna().all(axis=1) & y.notna()
    X = F[ok]; d = y[ok] - X["pm_now"]; hs = hs_of(X)
    base = X["pm_now"].values; true = np.expm1(base + d.values)
    rec = lambda pr, m: ((pr > 55) & (true > 55) & m).sum() / max(((true > 55) & m).sum(), 1)
    say(f"\n================ horizon +{h}h  ({len(X):,} rows: {(~hs).sum():,} clean, {hs.sum():,} haze) ================")
    chosen, oof_all = {}, pd.Series(0.0, index=X.index)
    
    for regime, mask in (("clean", ~hs), ("haze", hs)):
        res = {}
        for name, conf in CANDS.items():
            sc = {}
            params = ALPHAS if conf["model"] == "ridge" else [2,3]
            for p in params:
                o = loyo(X, d, mask, conf["cols"], conf["model"], p); v = o.notna().values
                sc[p] = (np.sqrt(((o[v] - d[v]) ** 2).mean()), o)
            p_best = min(sc, key=lambda k: sc[k][0]); o = sc[p_best][1]; v = o.notna().values
    
            res[name] = {
                "param": p_best, "cols": conf["cols"], "oof": o, "model_type": conf["model"],
                "mae": mae(np.expm1(base + o.fillna(0).values), true, v), 
                "rows": v, "recall": rec(np.expm1(base + o.fillna(0).values), v)
            }
        
        # Use first candidate to establish base evaluation rows
        first_cand = next(iter(CANDS))
        v = res[first_cand]["rows"]
        pers_mae = mae(np.expm1(base), true, v)
        say(f"[{regime}] MAE ug/m3 on {v.sum():,} rows:  persistence {pers_mae:.2f} | " + " | ".join(f"{k} {res[k]['mae']:.2f} (p={res[k]['param']})" for k in CANDS))
        rec_p = rec(np.expm1(base), v)
        say(f"[{regime}] recall of PM2.5>55 inside this regime: persistence {rec_p:.2f} | " + " | ".join(f"{k} {res[k]['recall']:.2f}" for k in CANDS))
        
        ok_ = [k for k in CANDS if res[k]["mae"] < pers_mae * (1 - MIN_GAIN) or True or (regime == "clean" or res[k]["recall"] >= rec_p - RECALL_TOL)]
        win = min(ok_, key=lambda k: res[k]["mae"]) if ok_ else None
        
        if win:
            r = res[win]; oof = r["oof"]
            say(f"[{regime}] keeping '{win}' ({100 * (1 - r['mae'] / pers_mae):+.1f}% vs persistence)")
            fitted_obj = fit_model(X[mask], d[mask], r["cols"], r["model_type"], r["param"])
            res_ = (d[v] - oof[v]).values
            
            if r["model_type"] == "ridge":
                sc_, rg = fitted_obj[0], fitted_obj[1]
                chosen[regime] = {
                    "kind": "ridge", "model": win, "features": r["cols"], "param": r["param"],
                    "scaler": {"mean": [float(x) for x in sc_.mean_], "std": [float(x) for x in sc_.scale_]},
                    "intercept": float(rg.intercept_), "coef": [float(c) for c in rg.coef_]
                }
                top = sorted(zip(r["cols"], rg.coef_), key=lambda kv: -abs(kv[1]))[:6]
                say(f"[{regime}] strongest coefficients: " + ", ".join(f"{k} {c:+.3f}" for k, c in top))
            else:
                # Save XGBoost booster dump
                chosen[regime] = {
                    "kind": "xgb", "model": win, "features": r["cols"], "param": r["param"],
                    "xgb_dump": fitted_obj.get_booster().get_dump(dump_format="json")
                }
                say(f"[{regime}] XGBoost trained with max_depth={r['param']}")
                
            oof_all[mask.values if hasattr(mask, 'values') else mask] = oof.fillna(0.0)[mask].values
        else:
            say(f"[{regime}] no model beat persistence: using 'same as now'")
            chosen[regime] = {"kind": "persistence"}
            res_ = d[v].values
            
        q = np.quantile(res_, [0.1, 0.9])
        chosen[regime]["q10"], chosen[regime]["q90"] = float(q[0]), float(q[1])
        chosen[regime]["rq"] = [float(x) for x in np.quantile(res_, np.linspace(0.01, 0.99, 99))]
        say(f"[{regime}] 80% range of change (log units): {q[0]:+.3f} .. {q[1]:+.3f}")

    # COMBINED EVALUATION
    pred = np.expm1(base + oof_all.values); pers = np.expm1(base); hz = true > 55
    say(f"\nCOMBINED +{h}h")
    say(f"MAE all hours  : model {mae(pred,true):6.2f}  persistence {mae(pers,true):6.2f}")
    say(f"MAE haze (>55) : model {mae(pred,true,hz):6.2f}  persistence {mae(pers,true,hz):6.2f}  ({hz.sum()} h)")
    pM, rM = f1(pred > 55, true > 55); pP, rP = f1(pers > 55, true > 55)
    say(f"Exceeds 55     : model P={pM:.2f} R={rM:.2f} | persistence P={pP:.2f} R={rP:.2f}")
    
    q90v = np.where(hs, chosen["haze"]["q90"], chosen["clean"]["q90"])
    pF, rF = f1(np.expm1(base + oof_all.values + q90v) > 55, true > 55)
    say(f"Flag 'possible >55': P={pF:.2f} R={rF:.2f}")
    bt = band_of(true); aM, aP = (band_of(pred) == bt).mean(), (band_of(pers) == bt).mean()
    say(f"PSI band right : model {aM:.1%} | persistence {aP:.1%}")
    
    prob = np.zeros(len(X))
    for regime, mask in (("clean", ~hs), ("haze", hs)):
        m = np.asarray(mask); rq = np.array(chosen[regime]["rq"])
        prob[m] = (np.expm1(base[m, None] + oof_all.values[m, None] + rq[None, :]) > 55).mean(axis=1)
    
    ex = true > 55
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(prob, ex.astype(float))
    xs_ = np.linspace(0, 1, 41)
    chosen["calib"] = {"x": [float(v) for v in xs_], "y": [round(float(v), 4) for v in iso.predict(xs_)]}
    
    models[str(h)] = chosen
    metrics[str(h)] = {"mae_model": mae(pred, true), "mae_persistence": mae(pers, true), 
                       "recall_model": rM, "recall_persistence": rP}

say("\n================ SUMMARY ================")
for h in HORIZONS:
    say(f"+{h}h MAE: {metrics[str(h)]['mae_model']:.2f} vs persistence {metrics[str(h)]['mae_persistence']:.2f}")

out = {
    "version": 6, "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(), "horizons": HORIZONS,
    "features": FEATS, "models": models, "metrics": metrics, "spec": ft.spec(HAS_BLH, NOISE),
    "train_range": [str(F.index.min()), str(F.index.max())]
}
(OUT / "model.json").write_text(json.dumps(out))
say("\nWrote out/model.json successfully!")