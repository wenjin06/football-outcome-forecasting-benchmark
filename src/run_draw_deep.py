"""
Deep draw analysis: why does the model almost never predict a draw?
==================================================================
Analyzing test 2025/26:
1. Model p_draw distribution (binned) vs. actual draw rate per bin
2. Market de-vig p_draw distribution vs. model p_draw
3. Class-wise calibration (per-class ECE for H/D/A)
4. Multiclass log-loss decomposed by class
5. Class priors (train/val/test)

Output: results/draw_deep.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss

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

inv = 1.0 / tmeta[["B365CH", "B365CD", "B365CA"]].replace(0, np.nan)
s = inv.sum(axis=1)
mkt_proba = (inv.div(s, axis=0)).values
valid = ~np.isnan(mkt_proba).any(axis=1)

results = {}

# Class priors
for name, part in [("train", train), ("val", val), ("test", test)]:
    y = part["y"].values.astype(int)
    results[f"prior_{name}"] = {"H": float((y == 0).mean()), "D": float((y == 1).mean()),
                                "A": float((y == 2).mean())}
print("priors:", {k: {kk: round(v, 3) for kk, v in vv.items()} for k, vv in results.items() if k.startswith("prior")})

# 1. Model p_draw distribution
pd_model = proba[:, 1]
bins = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]
rows = []
for i in range(len(bins) - 1):
    mask = (pd_model > bins[i]) & (pd_model <= bins[i + 1]) & valid
    if mask.sum() < 5:
        continue
    rows.append({"bin": f"({bins[i]},{bins[i+1]}]", "n": int(mask.sum()),
                 "mean_p": float(pd_model[mask].mean()),
                 "actual_draw_rate": float((yte[mask] == 1).mean())})
results["model_pdraw_bins"] = rows
print("\nmodel p_draw bins:")
for r in rows:
    print(f"  {r['bin']}: n={r['n']} mean_p={r['mean_p']:.3f} actual={r['actual_draw_rate']*100:.1f}%")

# 2. Market p_draw distribution
pd_mkt = mkt_proba[:, 1]
rows2 = []
for i in range(len(bins) - 1):
    mask = (pd_mkt > bins[i]) & (pd_mkt <= bins[i + 1]) & valid
    if mask.sum() < 5:
        continue
    rows2.append({"bin": f"({bins[i]},{bins[i+1]}]", "n": int(mask.sum()),
                  "mean_p": float(pd_mkt[mask].mean()),
                  "actual_draw_rate": float((yte[mask] == 1).mean())})
results["market_pdraw_bins"] = rows2
print("\nmarket de-vig p_draw bins:")
for r in rows2:
    print(f"  {r['bin']}: n={r['n']} mean_p={r['mean_p']:.3f} actual={r['actual_draw_rate']*100:.1f}%")

# 3. Class-wise calibration ECE (each class independently)
def class_ece(y, p_class, cls, n_bins=10):
    conf = p_class
    acc = (y == cls).astype(float)
    bins_e = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins_e[i]) & (conf <= bins_e[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y)) * abs(acc[m].mean() - conf[m].mean())
    return ece

maskv = valid
results["class_ece"] = {
    "H": float(class_ece(yte[maskv], proba[maskv, 0], 0)),
    "D": float(class_ece(yte[maskv], proba[maskv, 1], 1)),
    "A": float(class_ece(yte[maskv], proba[maskv, 2], 2)),
    "market_H": float(class_ece(yte[maskv], mkt_proba[maskv, 0], 0)),
    "market_D": float(class_ece(yte[maskv], mkt_proba[maskv, 1], 1)),
    "market_A": float(class_ece(yte[maskv], mkt_proba[maskv, 2], 2)),
}
print("\nper-class ECE:", {k: round(v, 4) for k, v in results["class_ece"].items()})

# 4. Multiclass log-loss by class: mean -log p(true class) per class
pm = proba[maskv][np.arange(maskv.sum()), yte[maskv]]
ll_by_class = {}
for c in range(3):
    m = (yte[maskv] == c)
    if m.sum() > 0:
        ll_by_class[str(c)] = float(-np.log(pm[m]).mean())
results["logloss_by_class"] = ll_by_class
print("\nper-class log-loss:", {k: round(v, 4) for k, v in ll_by_class.items()})

# 5. Correlations: p_draw vs. actual draw
results["corr_pdraw_draw"] = float(np.corrcoef(pd_model[maskv], (yte[maskv] == 1).astype(float))[0, 1])
results["corr_mktpdraw_draw"] = float(np.corrcoef(pd_mkt[maskv], (yte[maskv] == 1).astype(float))[0, 1])
print("\ncorr(model p_draw, draw):", round(results["corr_pdraw_draw"], 3))
print("corr(market p_draw, draw):", round(results["corr_mktpdraw_draw"], 3))

# 6. Brier decomposition (per-class binary Murphy decomposition): BS = Uncertainty - Resolution + Reliability
#    applied separately to each class (class probability vs. class occurrence), model and market side by side
print("\n[6] Brier decomposition (per class)...")

def brier_decomp_binary(y_bin, f, n_bins=10):
    """f: predicted probability array; y_bin: 0/1 labels. Returns Murphy decomposition components."""
    n = len(y_bin)
    o_bar = y_bin.mean()
    bs = float(np.mean((f - y_bin) ** 2))
    unc = float(o_bar * (1 - o_bar))
    bins = np.linspace(0, 1, n_bins + 1)
    rel = 0.0
    res = 0.0
    for i in range(n_bins):
        m = (f > bins[i]) & (f <= bins[i + 1])
        if m.sum() == 0:
            continue
        fm = f[m].mean()
        om = y_bin[m].mean()
        rel += (m.sum() / n) * (fm - om) ** 2
        res += (m.sum() / n) * (om - o_bar) ** 2
    return {"brier": bs, "uncertainty": unc, "reliability": float(rel),
            "resolution": float(res)}

brier_decomp = {}
for c, cls in [(0, "H"), (1, "D"), (2, "A")]:
    yb = (yte[maskv] == c).astype(float)
    brier_decomp[cls] = {
        "model": brier_decomp_binary(yb, proba[maskv, c]),
        "market": brier_decomp_binary(yb, mkt_proba[maskv, c]),
    }
    print(f"  {cls}: model BS={brier_decomp[cls]['model']['brier']:.4f} "
          f"U={brier_decomp[cls]['model']['uncertainty']:.4f} "
          f"Rel={brier_decomp[cls]['model']['reliability']:.4f} "
          f"Res={brier_decomp[cls]['model']['resolution']:.4f} | "
          f"market BS={brier_decomp[cls]['market']['brier']:.4f} "
          f"Res={brier_decomp[cls]['market']['resolution']:.4f}")
results["brier_decomp"] = brier_decomp

with open(os.path.join(RES, "draw_deep.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("\nsaved:", os.path.join(RES, "draw_deep.json"))
