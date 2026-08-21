"""
Error analysis: failed predictions vs. odds movement / market disagreement
==========================================================================
All questions are answered on the same protocol as the main experiments
(test 2025/26 first half, 1,104 matches):

1. Model-market disagreement: when the XGB argmax differs from the de-vigged
   market argmax, which side is right more often? (direct evidence on whether
   the model can beat the market where they disagree - RQ1 market efficiency)
2. Model error rate by odds-movement (closing-opening) buckets: are matches
   with large movement harder to predict?
3. Error rate by market-disagreement (dispersion across bookmakers) buckets:
   are matches with high bookmaker disagreement harder to predict?
4. High-confidence errors: share of wrong predictions among high-confidence
   (max prob >= 0.6) predictions, vs. the market under the same criterion
   (overconfidence evidence, consistent with the value-betting findings)
5. Draw focus: whether misclassified draws differ from correctly classified
   draws in odds movement / dispersion / confidence

Output: results/error_analysis.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

OUT = r"E:\论文\sci_redo\data\processed"
RES = r"E:\论文\sci_redo\results"
RAW_GLOB = r"E:\论文\structured_data\*.csv"
os.makedirs(RES, exist_ok=True)

# ---------------- Data loading (same protocol as run_value_betting.py) ----------------
feat = pd.read_csv(os.path.join(OUT, "all_matches_featurized.csv"), parse_dates=["Date"])
drop_cols = ["Div", "Date", "Season", "HomeTeam", "AwayTeam", "FTR", "y"]
feature_cols = [c for c in feat.columns if c not in drop_cols and feat[c].notna().sum() > 0]
medians = feat[feat["Date"] < "2024-08-01"][feature_cols].median()
feat[feature_cols] = feat[feature_cols].fillna(medians)

train = feat[feat["Date"] < "2024-08-01"]
val = feat[(feat["Date"] >= "2024-08-01") & (feat["Date"] < "2025-08-01")]
test = feat[feat["Date"] >= "2025-08-01"]
yte = test["y"].values.astype(int)

raw = pd.concat([pd.read_csv(p) for p in glob.glob(RAW_GLOB)], ignore_index=True)
raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
tmeta = test.merge(raw[["Date", "HomeTeam", "AwayTeam",
                        "B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA",
                        "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA",
                        "MaxCH", "MaxCD", "MaxCA", "AvgCH", "AvgCD", "AvgCA"]],
                   on=["Date", "HomeTeam", "AwayTeam"], how="left")


def devig(h, d, a):
    inv = 1.0 / pd.concat([h, d, a], axis=1).replace(0, np.nan)
    s = inv.sum(axis=1)
    return (inv.div(s, axis=0)).values


# ---------------- Model (same parameters as the main experiments) ----------------
print("[1] training global XGB ...")
model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                      early_stopping_rounds=30, random_state=42)
model.fit(train[feature_cols], train["y"].values.astype(int),
          eval_set=[(val[feature_cols], val["y"].values.astype(int))], verbose=False)
proba = model.predict_proba(test[feature_cols])
pred = proba.argmax(axis=1)
err = (pred != yte).astype(float)

market_proba = devig(tmeta["B365CH"], tmeta["B365CD"], tmeta["B365CA"])
mkt_pred = np.nanargmax(market_proba, axis=1)
mkt_valid = ~np.isnan(market_proba).any(axis=1)
mkt_err = (mkt_pred != yte).astype(float)

n = len(yte)
results = {"meta": {"n_test": int(n), "model": "XGBoost(v3, xG)",
                    "test_window": "2025/26 through 2026-02-12"}}

# ---------------- 1. Model-market disagreement: who is right more often ----------------
print("\n[2] model vs market: disagreement matches ...")
valid = mkt_valid
agree = (pred == mkt_pred) & valid
disagree = (pred != mkt_pred) & valid
acc = lambda e: 1.0 - e.mean()

results["market_vs_model"] = {
    "n_agree": int(agree.sum()), "n_disagree": int(disagree.sum()),
    "acc_model_overall": acc(err[valid]), "acc_market_overall": acc(mkt_err[valid]),
    "acc_model_on_agree": acc(err[agree]), "acc_market_on_agree": acc(mkt_err[agree]),
    "acc_model_on_disagree": acc(err[disagree]), "acc_market_on_disagree": acc(mkt_err[disagree]),
    "model_win_disagree": float(acc(err[disagree]) - acc(mkt_err[disagree])),
}
print(f"  overall acc: model {acc(err[valid]):.3f} / market {acc(mkt_err[valid]):.3f}")
print(f"  disagreement {int(disagree.sum())} matches: model {acc(err[disagree]):.3f} / market {acc(mkt_err[disagree]):.3f}")
print(f"  agreement {int(agree.sum())} matches: model {acc(err[agree]):.3f} / market {acc(mkt_err[agree]):.3f}")

# ---------------- 2. Odds-movement buckets (closing-opening, B365) ----------------
print("\n[3] odds-movement buckets ...")
move_h = tmeta["B365CH"] - tmeta["B365H"]
move_d = tmeta["B365CD"] - tmeta["B365D"]
move_a = tmeta["B365CA"] - tmeta["B365A"]
move = (move_h.abs() + move_d.abs() + move_a.abs()) / 3.0
# Confounding diagnostics: favorite share (min opening odds <= 1.6) and market max de-vig probability
init_odds = tmeta[["B365H", "B365D", "B365A"]]
fav_share = (init_odds.min(axis=1) <= 1.6).astype(float)
mkt_max_prob = np.nanmax(market_proba, axis=1)


def ci_error_rate(e, seed):
    rng = np.random.default_rng(seed)
    boot = np.array([e[rng.integers(0, len(e), len(e))].mean() for _ in range(2000)])
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


move_bins = pd.qcut(move, 5, labels=False, duplicates="drop")
move_rows = []
for b in range(int(move_bins.max()) + 1):
    mask = (move_bins == b) & move.notna()
    if mask.sum() == 0:
        continue
    e = err[mask]
    move_rows.append({"bin": int(b), "move_range": f"[{move[mask].min():.3f},{move[mask].max():.3f}]",
                      "n": int(mask.sum()), "error_rate": float(e.mean()),
                      "error_rate_ci": ci_error_rate(np.asarray(e, dtype=float), 42 + b),
                      "fav_share": float(fav_share[mask].mean()),
                      "market_max_prob": float(mkt_max_prob[mask].mean())})
results["odds_movement_bins"] = move_rows
for r in move_rows:
    print(f"  movement bucket{r['bin']} {r['move_range']}: n={r['n']} error rate={r['error_rate']*100:.1f}% "
          f"fav share={r['fav_share']*100:.0f}% market maxP={r['market_max_prob']:.2f}")

# ---------------- 3. Market-disagreement buckets (bookmaker dispersion, closing) ----------------
print("\n[4] market-dispersion buckets ...")
disp = ((tmeta["MaxCH"] - tmeta["AvgCH"]) / tmeta["AvgCH"].replace(0, np.nan) +
        (tmeta["MaxCD"] - tmeta["AvgCD"]) / tmeta["AvgCD"].replace(0, np.nan) +
        (tmeta["MaxCA"] - tmeta["AvgCA"]) / tmeta["AvgCA"].replace(0, np.nan)) / 3.0
disp_bins = pd.qcut(disp, 5, labels=False, duplicates="drop")
disp_rows = []
for b in range(int(disp_bins.max()) + 1):
    mask = (disp_bins == b) & disp.notna()
    if mask.sum() == 0:
        continue
    e = err[mask]
    disp_rows.append({"bin": int(b), "disp_range": f"[{disp[mask].min():.4f},{disp[mask].max():.4f}]",
                      "n": int(mask.sum()), "error_rate": float(e.mean()),
                      "error_rate_ci": ci_error_rate(np.asarray(e, dtype=float), 100 + b),
                      "fav_share": float(fav_share[mask].mean()),
                      "market_max_prob": float(mkt_max_prob[mask].mean())})
results["market_dispersion_bins"] = disp_rows
for r in disp_rows:
    print(f"  dispersion bucket{r['bin']} {r['disp_range']}: n={r['n']} error rate={r['error_rate']*100:.1f}% "
          f"fav share={r['fav_share']*100:.0f}% market maxP={r['market_max_prob']:.2f}")

# ---------------- 4. High-confidence errors (overconfidence evidence) ----------------
print("\n[5] high-confidence errors ...")
conf = proba.max(axis=1)
mkt_conf = market_proba.max(axis=1)
for thr in [0.5, 0.6, 0.7]:
    hm = conf >= thr
    mm = (mkt_conf >= thr) & valid
    results[f"high_conf_{int(thr*100)}"] = {
        "model_n": int(hm.sum()), "model_error_rate": float(err[hm].mean()),
        "market_n": int(mm.sum()), "market_error_rate": float(mkt_err[mm].mean()),
    }
    print(f"  conf>={thr}: model n={int(hm.sum())} error rate={err[hm].mean()*100:.1f}% "
          f"| market n={int(mm.sum())} error rate={mkt_err[mm].mean()*100:.1f}%")

# ---------------- 5. Draw-specific errors ----------------
print("\n[6] draw misclassification analysis ...")
draw_true = yte == 1
draw_correct = draw_true & (pred == 1)
draw_wrong = draw_true & (pred != 1)
if draw_wrong.sum() > 0 and draw_correct.sum() > 0:
    def agg(mask):
        return {"n": int(mask.sum()),
                "mean_move": float(move[mask].mean()),
                "mean_disp": float(disp[mask].mean()),
                "mean_conf": float(conf[mask].mean())}
    results["draw_errors"] = {"draw_true_n": int(draw_true.sum()),
                              "correctly_predicted": agg(draw_correct),
                              "misclassified": agg(draw_wrong)}
    print(f"  true draws {int(draw_true.sum())}: correct {int(draw_correct.sum())} / misclassified {int(draw_wrong.sum())}")
    print(f"  misclassified: movement {move[draw_wrong].mean():.3f} dispersion {disp[draw_wrong].mean():.4f} confidence {conf[draw_wrong].mean():.3f}")
    print(f"  correct: movement {move[draw_correct].mean():.3f} dispersion {disp[draw_correct].mean():.4f} confidence {conf[draw_correct].mean():.3f}")

# ---------------- Save ----------------
with open(os.path.join(RES, "error_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("\nsaved:", os.path.join(RES, "error_analysis.json"))
