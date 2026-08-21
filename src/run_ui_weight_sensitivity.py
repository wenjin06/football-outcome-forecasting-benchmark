"""
UI weight sensitivity (revision comment 2a: the weights are fixed presets and
must be shown to be insensitive to small parameter changes)
====================
Test several UI weight sets on the validation set and report the tiering
behavior under each (low-tier acc/ROI, number of no-bet matches).
If the tiering pattern is consistent across weight sets (low tier best, high
tier worst), the conclusion is robust and not an artifact of parameter fitting.
Output: results/ui_weight_sensitivity.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from evaluate import financial_metrics, simulate_bets
from risk import compute_ui, fit_robust, risk_tiers

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

raw = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(paths.raw_data_dir(), "*.csv"))], ignore_index=True)
raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
vmeta = val.merge(raw[["Date", "HomeTeam", "AwayTeam", "B365CH", "B365CD", "B365CA"]],
                  on=["Date", "HomeTeam", "AwayTeam"], how="left")

xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                    early_stopping_rounds=30, random_state=42)
xgb.fit(train[feature_cols], train["y"].values.astype(int),
        eval_set=[(val[feature_cols], val["y"].values.astype(int))], verbose=False)
proba = xgb.predict_proba(val[feature_cols])
yva = val["y"].values.astype(int)
odds = vmeta[["B365CH", "B365CD", "B365CA"]].values

vol_stats = fit_robust(train, "close_vol")
move_stats = fit_robust(train, "odds_move_H")

WEIGHTS = {
    "w=(0.40,0.30,0.15,0.15)": (0.40, 0.30, 0.15, 0.15),
    "w=(0.50,0.20,0.15,0.15)": (0.50, 0.20, 0.15, 0.15),
    "w=(0.30,0.40,0.15,0.15)": (0.30, 0.40, 0.15, 0.15),
    "w=(0.40,0.30,0.20,0.10)": (0.40, 0.30, 0.20, 0.10),
    "w=(0.40,0.30,0.10,0.20)": (0.40, 0.30, 0.10, 0.20),
    "w=(0.35,0.25,0.20,0.20)": (0.35, 0.25, 0.20, 0.20),
}

results = {}
for name, w in WEIGHTS.items():
    ui = compute_ui(proba, val, vol_stats, move_stats, w=w)
    tier, _ = risk_tiers(ui, 0.30, 0.45)
    rows = {}
    for t, lab in [(0, "low"), (1, "medium"), (2, "high(no-bet)")]:
        mask = tier == t
        if mask.sum() == 0:
            rows[lab] = {"n": 0}
            continue
        acc = accuracy_score(yva[mask], proba[mask].argmax(axis=1))
        rets = np.zeros(mask.sum()); placed = np.zeros(mask.sum(), dtype=bool)
        sub_p, sub_o, sub_y = proba[mask], odds[mask], yva[mask]
        for i in range(mask.sum()):
            j = int(sub_p[i].argmax())
            o = sub_o[i, j]
            if np.isfinite(o) and o > 1:
                placed[i] = True
                rets[i] = (o - 1) if j == sub_y[i] else -1.0
        fin = financial_metrics(rets, placed)
        rows[lab] = {"n": int(mask.sum()), "acc": acc, "roi": fin["roi"]}
    results[name] = rows
    print(f"{name}: low acc={rows['low']['acc']:.3f} roi={rows['low']['roi']*100:.2f}% "
          f"| no-bet n={rows['high(no-bet)'].get('n',0)} "
          f"roi={rows['high(no-bet)'].get('roi')}")

with open(os.path.join(RES, "ui_weight_sensitivity.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("saved:", os.path.join(RES, "ui_weight_sensitivity.json"))
