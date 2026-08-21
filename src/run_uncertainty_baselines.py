"""
Uncertainty-measure comparison: 1-pmax / entropy / 1-Sum p^2 (expected Brier) / UI
==================================================================================
Addresses the core question: why use the UI instead of standard uncertainty measures?

Protocol (test 2025/26, XGB probabilities, B365 closing odds, equal stakes):
1. Compute four measures per match:
   - 1-pmax   : inverse confidence (maximum model probability)
   - entropy  : H(p) = -sum p log p (standard 3-class entropy)
   - exp_brier: 1 - sum p^2 (expected Brier contribution, proper-scoring-rule motivation)
   - UI       : the uncertainty index proposed in the paper (includes market-volatility terms)
2. Stratification comparison:
   - Accuracy by tertile of each score (low/medium/high)
   - ROI of no-bet policies at coverage 0.7/0.5 (drop highest-risk matches, same protocol as Table 19)
   - Correlations between scores (is the UI merely a linear function of pmax/entropy?)
3. Statistical test: bootstrap test of the low-tier vs. high-tier accuracy difference

Output: results/uncertainty_baselines.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss as sk_logloss

from evaluate import financial_metrics, ece, brier_multiclass
from risk import fit_robust, compute_ui

import paths
OUT = paths.PROCESSED
RES = paths.RES
os.makedirs(RES, exist_ok=True)

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

print("[1] training global XGB ...")
model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                      early_stopping_rounds=30, random_state=42)
model.fit(train[feature_cols], train["y"].values.astype(int),
          eval_set=[(val[feature_cols], val["y"].values.astype(int))], verbose=False)
proba = model.predict_proba(test[feature_cols])
pred = proba.argmax(axis=1)
odds = tmeta[["B365CH", "B365CD", "B365CA"]].values

# Per-match returns (all-bets baseline, same protocol as run_policy_comparison.py)
rets = np.zeros(len(yte))
for i in range(len(yte)):
    o = odds[i]
    j = pred[i]
    if np.isfinite(o[j]) and o[j] > 1:
        rets[i] = (o[j] - 1) if j == yte[i] else -1.0
    else:
        rets[i] = np.nan
valid = ~np.isnan(rets)

# ---- Four uncertainty measures (unified direction: higher = more uncertain) ----
pmax = proba.max(axis=1)
eps = 1e-12
entropy = -np.sum(proba * np.log(np.clip(proba, eps, 1.0)), axis=1) / np.log(3.0)  # normalized to [0,1]
exp_brier = 1.0 - np.sum(proba ** 2, axis=1)

vol_stats = fit_robust(train, "close_vol")
move_stats = fit_robust(train, "odds_move_H")
ui = compute_ui(proba, test, vol_stats, move_stats)

scores = {
    "1-pmax": 1.0 - pmax,
    "entropy": entropy,
    "1-Sum p^2": exp_brier,
    "UI": ui,
}

# ---- Correlations between scores ----
corr = {}
for a in scores:
    for b in scores:
        if a < b:
            corr[f"{a} vs {b}"] = float(np.corrcoef(scores[a], scores[b])[0, 1])
print("\ncorrelations between scores:")
for k, v in corr.items():
    print(f"  {k}: {v:.3f}")

# ---- 1. Accuracy by tertile ----
tiers = {}
for name, sc in scores.items():
    q1, q2 = np.nanquantile(sc, [1 / 3, 2 / 3])
    t = np.where(sc <= q1, 0, np.where(sc <= q2, 1, 2))
    rows = []
    for ti, tname in [(0, "low"), (1, "medium"), (2, "high")]:
        mask = t == ti
        acc = accuracy_score(yte[mask], pred[mask])
        rows.append({"tier": tname, "n": int(mask.sum()), "acc": acc})
    tiers[name] = rows
    print(f"  {name}: " + ", ".join(f"{r['tier']}={r['acc']:.3f}" for r in rows))

# Bootstrap test of the low- vs. high-tier accuracy difference (independent samples; bootstrap the difference directly)
diff_boot = {}
for name, sc in scores.items():
    q1, q2 = np.nanquantile(sc, [1 / 3, 2 / 3])
    lo = sc <= q1
    hi = sc > q2
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(2000):
        i1 = rng.integers(0, lo.sum(), lo.sum())
        i2 = rng.integers(0, hi.sum(), hi.sum())
        a1 = (pred[lo][i1] == yte[lo][i1]).mean()
        a2 = (pred[hi][i2] == yte[hi][i2]).mean()
        diffs.append(a1 - a2)
    diffs = np.array(diffs)
    diff_boot[name] = {"diff": float(diffs.mean()),
                       "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                       "p_below_zero": float((diffs < 0).mean())}
print("\nlow-high tier accuracy difference (bootstrap 95% CI, p=Pr(diff<0)):")
for k, v in diff_boot.items():
    print(f"  {k}: {v['diff']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] p={v['p_below_zero']:.3f}")

# ---- 2. No-bet policies at coverage 0.7/0.5 (same protocol as Table 19) ----
def evaluate_at_coverage(risk, cov_frac):
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
            "acc": float(acc), "sharpe": fin["sharpe"], "mdd": fin["mdd"],
            "win_rate": fin["win_rate"],
            "avg_odds": float(np.nanmean(odds[mask][np.arange(mask.sum()), pred[mask]]))}

policies = {}
for name, sc in scores.items():
    policies[name] = [evaluate_at_coverage(sc, c) for c in (0.7, 0.5)]
    print(f"\n=== {name} ===")
    for r in policies[name]:
        print(f"  cov={r['coverage']}: n={r['n']} acc={r['acc']*100:.1f}% "
              f"ROI={r['roi']*100:.2f}% [{r['roi_ci'][0]*100:.1f},{r['roi_ci'][1]*100:.1f}] "
              f"Sharpe={r['sharpe']:.3f} avg_odds={r['avg_odds']:.2f}")

# ---- 3. Full-measure prediction metrics (for citation in the text) ----
metrics = {}
for name, sc in scores.items():
    q1, q2 = np.nanquantile(sc, [1 / 3, 2 / 3])
    t = np.where(sc <= q1, 0, np.where(sc <= q2, 1, 2))
    lo = t == 0
    metrics[name] = {
        "acc_low_tier": float(accuracy_score(yte[lo], pred[lo])),
        "n_low_tier": int(lo.sum()),
        "acc_high_tier": float(accuracy_score(yte[t == 2], pred[t == 2])),
        "n_high_tier": int((t == 2).sum()),
        "corr_with_UI": float(np.corrcoef(sc, ui)[0, 1]),
    }

results = {
    "correlations": corr,
    "tier_stratification": tiers,
    "tier_diff_bootstrap": diff_boot,
    "policies": policies,
    "metrics": metrics,
}
with open(os.path.join(RES, "uncertainty_baselines.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("\nsaved:", os.path.join(RES, "uncertainty_baselines.json"))
