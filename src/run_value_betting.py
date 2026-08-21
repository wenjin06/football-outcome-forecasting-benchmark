"""
Value betting + opening/closing odds tests + Kelly staking
====================
Recommended enhancements #3 (Kelly/stake comparison) and #7 (complete betting
protocol), plus market-efficiency tests.

1. Value betting: bet only when EV = p_model * odds - 1 > threshold; sweep the threshold
2. Opening vs. closing: use de-vigged opening probabilities as the model and settle
   at closing odds -> tests the information content of opening prices
3. Kelly: f = (p*odds-1)/(odds-1), capped at 10% stake, compared with equal stakes
Output: results/value_betting.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from evaluate import financial_metrics

OUT = r"E:\论文\sci_redo\data\processed"
RES = r"E:\论文\sci_redo\results"
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

raw = pd.concat([pd.read_csv(p) for p in glob.glob(r"E:\论文\structured_data\*.csv")], ignore_index=True)
raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
tmeta = test.merge(raw[["Date", "HomeTeam", "AwayTeam",
                        "B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA"]],
                   on=["Date", "HomeTeam", "AwayTeam"], how="left")


def devig(h, d, a):
    inv = 1.0 / pd.concat([h, d, a], axis=1).replace(0, np.nan)
    s = inv.sum(axis=1)
    return (inv.div(s, axis=0)).values


print("[1] training global XGB ...")
model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                      early_stopping_rounds=30, random_state=42)
model.fit(train[feature_cols], train["y"].values.astype(int),
          eval_set=[(val[feature_cols], val["y"].values.astype(int))], verbose=False)
proba = model.predict_proba(test[feature_cols])

closing = tmeta[["B365CH", "B365CD", "B365CA"]].values
opening_proba = devig(tmeta["B365H"], tmeta["B365D"], tmeta["B365A"])

results = {}

# ============ 1. Value betting (XGB probabilities vs. closing odds) ============
print("\n[2] value-betting threshold scan (EV = p*odds - 1)...")
ev_scan = []
for thr in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]:
    rets = np.zeros(len(yte))
    placed = np.zeros(len(yte), dtype=bool)
    for i in range(len(yte)):
        p = proba[i]
        odds = closing[i]
        evs = p * odds - 1
        j = int(np.argmax(evs))
        if evs[j] > thr and np.isfinite(odds[j]) and odds[j] > 1:
            placed[i] = True
            rets[i] = (odds[j] - 1) if j == yte[i] else -1.0
    fin = financial_metrics(rets, placed)
    ev_scan.append({"threshold": thr, "roi": fin["roi"], "sharpe": fin["sharpe"],
                    "mdd": fin["mdd"], "n_bets": int(fin["n_bets"]),
                    "win_rate": fin["win_rate"],
                    "coverage": float(placed.sum() / len(yte))})
    print(f"  EV>{thr:.2f}: ROI={fin['roi']*100:.2f}% Sharpe={fin['sharpe']:.3f} "
          f"MDD={fin['mdd']*100:.1f}% n={fin['n_bets']}")
results["value_ev_scan"] = ev_scan

# ============ 2. Opening vs. closing (intra-market efficiency) ============
print("\n[3] opening de-vig probabilities vs closing-odds settlement ...")
rets = np.zeros(len(yte))
placed = np.zeros(len(yte), dtype=bool)
pred_open = opening_proba.argmax(axis=1)
for i in range(len(yte)):
    odds = closing[i]
    j = pred_open[i]
    if np.isfinite(odds[j]) and odds[j] > 1:
        placed[i] = True
        rets[i] = (odds[j] - 1) if j == yte[i] else -1.0
fin = financial_metrics(rets, placed)
results["open_vs_close"] = {"roi": fin["roi"], "sharpe": fin["sharpe"], "mdd": fin["mdd"],
                            "n_bets": int(fin["n_bets"]), "win_rate": fin["win_rate"],
                            "accuracy": float((pred_open == yte).mean())}
print(f"  opening probs @ closing settlement: ROI={fin['roi']*100:.2f}% acc={(pred_open==yte).mean():.3f}")

# Reverse direction: closing de-vig probabilities settled at opening odds
print("\n[4] closing de-vig probabilities vs opening-odds settlement (direction check)...")
opening = tmeta[["B365H", "B365D", "B365A"]].values
closing_proba = devig(tmeta["B365CH"], tmeta["B365CD"], tmeta["B365CA"])
pred_close = closing_proba.argmax(axis=1)
rets = np.zeros(len(yte))
placed = np.zeros(len(yte), dtype=bool)
for i in range(len(yte)):
    odds = opening[i]
    j = pred_close[i]
    if np.isfinite(odds[j]) and odds[j] > 1:
        placed[i] = True
        rets[i] = (odds[j] - 1) if j == yte[i] else -1.0
fin = financial_metrics(rets, placed)
results["close_vs_open"] = {"roi": fin["roi"], "sharpe": fin["sharpe"], "mdd": fin["mdd"],
                            "n_bets": int(fin["n_bets"]), "win_rate": fin["win_rate"]}
print(f"  closing probs @ opening settlement: ROI={fin['roi']*100:.2f}%")

# ============ 3. Kelly staking vs. equal stakes ============
print("\n[5] Kelly staking (capped at 10%)...")
bankroll = 100.0
kelly_rets = np.zeros(len(yte))
kelly_placed = np.zeros(len(yte), dtype=bool)
for i in range(len(yte)):
    p = proba[i]
    odds = closing[i]
    j = int(np.argmax(p * odds - 1))
    ev = p[j] * odds[j] - 1
    if ev <= 0 or not (np.isfinite(odds[j]) and odds[j] > 1):
        continue
    f = (p[j] * odds[j] - 1) / (odds[j] - 1)
    f = min(max(f, 0), 0.10)
    stake = f * bankroll
    kelly_placed[i] = True
    if j == yte[i]:
        ret = stake * (odds[j] - 1)
    else:
        ret = -stake
    bankroll += ret
    kelly_rets[i] = ret
fin = financial_metrics(kelly_rets, kelly_placed)
results["kelly"] = {"roi": fin["roi"], "sharpe": fin["sharpe"], "mdd": fin["mdd"],
                    "n_bets": int(fin["n_bets"]), "win_rate": fin["win_rate"],
                    "final_bankroll": float(bankroll)}
print(f"  Kelly: ROI={fin['roi']*100:.2f}% Sharpe={fin['sharpe']:.3f} "
      f"MDD={fin['mdd']*100:.1f}% final bankroll={bankroll:.1f} ({fin['n_bets']} bets)")

# ============ 4. Value betting after calibration (test of the overconfidence hypothesis) ============
print("\n[6] EV scan redone after probability calibration (isotonic, fit on val)...")
from sklearn.isotonic import IsotonicRegression
proba_va = model.predict_proba(val[feature_cols])
yva = val["y"].values.astype(int)
calib = []
for c in range(3):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba_va[:, c], (yva == c).astype(float))
    calib.append(iso)
proba_cal = np.column_stack([calib[c].predict(proba[:, c]) for c in range(3)])
proba_cal = proba_cal / proba_cal.sum(axis=1, keepdims=True)

cal_scan = []
for thr in [0.0, 0.02, 0.05, 0.08, 0.10]:
    rets = np.zeros(len(yte))
    placed = np.zeros(len(yte), dtype=bool)
    for i in range(len(yte)):
        p = proba_cal[i]
        odds = closing[i]
        evs = p * odds - 1
        j = int(np.argmax(evs))
        if evs[j] > thr and np.isfinite(odds[j]) and odds[j] > 1:
            placed[i] = True
            rets[i] = (odds[j] - 1) if j == yte[i] else -1.0
    fin = financial_metrics(rets, placed)
    cal_scan.append({"threshold": thr, "roi": fin["roi"], "sharpe": fin["sharpe"],
                     "mdd": fin["mdd"], "n_bets": int(fin["n_bets"]),
                     "win_rate": fin["win_rate"],
                     "coverage": float(placed.sum() / len(yte))})
    print(f"  after calibration EV>{thr:.2f}: ROI={fin['roi']*100:.2f}% n={fin['n_bets']}")
results["value_ev_scan_calibrated"] = cal_scan

with open(os.path.join(RES, "value_betting.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("\nsaved:", os.path.join(RES, "value_betting.json"))
