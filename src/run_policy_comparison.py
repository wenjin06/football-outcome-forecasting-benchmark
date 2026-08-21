"""
Policy-level comparison: UI vs. market-confidence vs. SCS vs. model-confidence vs. random
=========================================================================================
Addresses the core question: why not simply stratify by market max probability
and no-bet accordingly?

Unified protocol (test 2025/26, XGB probabilities, B365 closing odds, equal stakes):
- Each policy has a risk score; matches are removed (no-bet) from highest score
  downward, sweeping coverage
- ROI/Sharpe/MDD/acc are compared at the same coverage; coverage-ROI and
  coverage-acc curves are plotted
- If the UI curve lies above market-conf-only, the UI provides incremental
  risk stratification

Output: results/policy_comparison.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from risk import fit_robust, compute_ui, compute_scs
from evaluate import financial_metrics

import paths
OUT = paths.PROCESSED
RES = paths.RES

feat = pd.read_csv(os.path.join(OUT, "all_matches_featurized.csv"), parse_dates=["Date"])
drop_cols = ["Div", "Date", "Season", "HomeTeam", "AwayTeam", "FTR", "y"]
feature_cols = [c for c in feat.columns if c not in drop_cols and feat[c].notna().sum() > 0]
medians = feat[feat["Date"] < "2024-08-01"][feature_cols].median()
feat[feature_cols] = feat[feature_cols].fillna(medians)

train = feat[feat["Date"] < "2024-08-01"]
val = feat[(feat["Date"] >= "2024-08-01") & (feat["Date"] < "2025-08-01")]
test = feat[feat["Date"] >= "2025-08-01"]
yte = test["y"].values.astype(int)

raw = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(paths.raw_data_dir(), "*.csv"))], ignore_index=True)
raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
tmeta = test.merge(raw[["Date", "HomeTeam", "AwayTeam", "B365CH", "B365CD", "B365CA"]],
                   on=["Date", "HomeTeam", "AwayTeam"], how="left")

model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                      early_stopping_rounds=30, random_state=42)
model.fit(train[feature_cols], train["y"].values.astype(int),
          eval_set=[(val[feature_cols], val["y"].values.astype(int))], verbose=False)
proba = model.predict_proba(test[feature_cols])
pred = proba.argmax(axis=1)
odds = tmeta[["B365CH", "B365CD", "B365CA"]].values

# Per-match returns (all-bets baseline)
rets = np.zeros(len(yte))
for i in range(len(yte)):
    o = odds[i]
    j = pred[i]
    if np.isfinite(o[j]) and o[j] > 1:
        rets[i] = (o[j] - 1) if j == yte[i] else -1.0
    else:
        rets[i] = np.nan
valid = ~np.isnan(rets)

# ---- Risk scores ----
inv = 1.0 / tmeta[["B365CH", "B365CD", "B365CA"]].replace(0, np.nan)
s = inv.sum(axis=1)
mkt_proba = (inv.div(s, axis=0)).values
mkt_max = np.nanmax(mkt_proba, axis=1)          # market confidence (higher = more certain)
model_max = proba.max(axis=1)                    # model confidence
eps = 1e-12
entropy = -np.sum(proba * np.log(np.clip(proba, eps, 1.0)), axis=1) / np.log(3.0)
exp_brier = 1.0 - np.sum(proba ** 2, axis=1)     # expected Brier contribution (proper-score motivation)

vol_stats = fit_robust(train, "close_vol")
move_stats = fit_robust(train, "odds_move_H")
ui = compute_ui(proba, test, vol_stats, move_stats)   # higher = more uncertain
scs = compute_scs(test, vol_stats, move_stats)        # higher = more uncertain
rng = np.random.default_rng(0)
rand = rng.random(len(yte))

# Unified risk score: higher = more uncertain -> no-bet first
scores = {
    "UI": ui,
    "SCS": scs,
    "entropy": entropy,          # standard predictive-uncertainty measure (reference)
    "exp_brier": exp_brier,      # standard predictive-uncertainty measure (reference)
    "market_conf": -mkt_max,     # negated: invert market certainty
    "model_conf": -model_max,    # negated model confidence
    "random": rand,
}


def evaluate_at_coverage(risk, cov_frac):
    """Drop the highest-risk (1-cov_frac) fraction of matches and evaluate the remaining bets. cov=1.0 means no filtering."""
    if cov_frac >= 1.0:
        mask = valid
    else:
        thr = np.nanquantile(risk[valid], 1 - cov_frac)
        mask = (risk <= thr) & valid
    if mask.sum() < 20:
        return None
    r = rets[mask]
    fin = financial_metrics(r)
    acc = (pred[mask] == yte[mask]).mean()
    # Bootstrap CIs (ROI and acc)
    rngb = np.random.default_rng(42)
    n = len(r)
    roi_boot, acc_boot = [], []
    for _ in range(2000):
        idx = rngb.integers(0, n, n)
        rb = r[idx]
        roi_boot.append(rb.mean())
        acc_boot.append((pred[mask][idx] == yte[mask][idx]).mean())
    return {"coverage": float(cov_frac), "n": int(mask.sum()),
            "roi": fin["roi"], "roi_ci": [float(np.percentile(roi_boot, 2.5)),
                                             float(np.percentile(roi_boot, 97.5))],
            "acc": float(acc), "acc_ci": [float(np.percentile(acc_boot, 2.5)),
                                            float(np.percentile(acc_boot, 97.5))],
            "sharpe": fin["sharpe"], "mdd": fin["mdd"],
            "win_rate": fin["win_rate"],
            "avg_odds": float(np.nanmean(odds[mask][np.arange(mask.sum()), pred[mask]]))}


coverages = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]
results = {"coverages": coverages, "strategies": {}}
for name, sc in scores.items():
    curve = []
    for c in coverages:
        r = evaluate_at_coverage(sc, c)
        if r:
            curve.append(r)
    results["strategies"][name] = curve
    print(f"=== {name} ===")
    for r in curve:
        print(f"  cov={r['coverage']:.1f}: n={r['n']} acc={r['acc']*100:.1f}% "
              f"ROI={r['roi']*100:.2f}% [{r['roi_ci'][0]*100:.1f},{r['roi_ci'][1]*100:.1f}] "
              f"Sharpe={r['sharpe']:.3f} avg_odds={r['avg_odds']:.2f}")

with open(os.path.join(RES, "policy_comparison.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("\nsaved:", os.path.join(RES, "policy_comparison.json"))
