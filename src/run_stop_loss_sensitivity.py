"""
Stop-loss parameter sensitivity (revision comment 4b: 5 consecutive losses and
10% drawdown are arbitrary settings and need a sensitivity analysis)
====================
Sweep (stop_loss_n, stop_loss_dd) combinations on the test set and report
ROI/Sharpe/MDD.
Output: results/stop_loss_sensitivity.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from evaluate import financial_metrics
from risk import compute_ui, fit_robust, simulate_bets_with_risk

OUT = r"E:\论文\sci_redo\data\processed"
RES = r"E:\论文\sci_redo\results"

feat = pd.read_csv(os.path.join(OUT, "all_matches_featurized.csv"), parse_dates=["Date"])
drop_cols = ["Div", "Date", "Season", "HomeTeam", "AwayTeam", "FTR", "y"]
feature_cols = [c for c in feat.columns if c not in drop_cols and feat[c].notna().sum() > 0]
medians = feat[feat["Date"] < "2024-08-01"][feature_cols].median()
feat[feature_cols] = feat[feature_cols].fillna(medians)

train = feat[feat["Date"] < "2024-08-01"]
val = feat[(feat["Date"] >= "2024-08-01") & (feat["Date"] < "2025-08-01")]
test = feat[feat["Date"] >= "2025-08-01"]

raw = pd.concat([pd.read_csv(p) for p in glob.glob(r"E:\论文\structured_data\*.csv")], ignore_index=True)
raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
tmeta = test.merge(raw[["Date", "HomeTeam", "AwayTeam", "B365CH", "B365CD", "B365CA"]],
                   on=["Date", "HomeTeam", "AwayTeam"], how="left")

xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                    early_stopping_rounds=30, random_state=42)
xgb.fit(train[feature_cols], train["y"].values.astype(int),
        eval_set=[(val[feature_cols], val["y"].values.astype(int))], verbose=False)
proba = xgb.predict_proba(test[feature_cols])
yte = test["y"].values.astype(int)
odds = tmeta[["B365CH", "B365CD", "B365CA"]].values

vol_stats = fit_robust(train, "close_vol")
move_stats = fit_robust(train, "odds_move_H")
ui = compute_ui(proba, test, vol_stats, move_stats)

CONFIGS = [
    ("no stop-loss", 10**6, 10.0),
    ("n=5, dd=10% (paper)", 5, 0.10),
    ("n=3, dd=5%", 3, 0.05),
    ("n=3, dd=10%", 3, 0.10),
    ("n=8, dd=10%", 8, 0.10),
    ("n=5, dd=5%", 5, 0.05),
    ("n=5, dd=20%", 5, 0.20),
    ("n=10, dd=15%", 10, 0.15),
]

results = {}
for name, n, dd in CONFIGS:
    rets, placed, _ = simulate_bets_with_risk(
        yte, proba, odds[:, 0], odds[:, 1], odds[:, 2], ui,
        t_low=0.30, t_hi=0.45, stop_loss_n=n, stop_loss_dd=dd)
    fin = financial_metrics(rets, placed)
    results[name] = {"roi": fin["roi"], "sharpe": fin["sharpe"], "mdd": fin["mdd"],
                     "win_rate": fin["win_rate"], "n_bets": int(fin["n_bets"])}
    print(f"{name}: ROI={fin['roi']*100:.2f}% Sharpe={fin['sharpe']:.3f} "
          f"MDD={fin['mdd']*100:.1f}% ({fin['n_bets']} bets)")

with open(os.path.join(RES, "stop_loss_sensitivity.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("saved:", os.path.join(RES, "stop_loss_sensitivity.json"))
