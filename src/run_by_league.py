"""
Per-league / per-season / leave-one-league-out generalization analysis
====================
Required revision #8 (per-league/per-season breakdown) and recommended enhancement #2 (LOLO).

1. Global XGB (same hyperparameters as the baselines) -> test broken down by league, val+test by season
2. Leave-one-league-out: train on the other four leagues' train+val, evaluate on the held-out league's test
Output: results/by_league.json
"""
import os
import json
import glob
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss

from evaluate import evaluate_predictions, financial_metrics, simulate_bets

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

# Test-set odds
raw = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(paths.raw_data_dir(), "*.csv"))], ignore_index=True)
raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])


def eval_group(name, X, y, meta, odds_df):
    proba = model.predict_proba(X)
    res = evaluate_predictions(y, proba)
    rets, placed = simulate_bets(y, proba, odds_df["B365CH"].values,
                                 odds_df["B365CD"].values, odds_df["B365CA"].values,
                                 min_prob=0.0)
    fin = financial_metrics(rets, placed)
    return {"n": int(len(y)), "accuracy": res["accuracy"], "macro_f1": res["macro_f1"],
            "log_loss": res["log_loss"], "brier": res["brier"], "ece": res["ece"],
            "roi": fin["roi"], "mdd": fin["mdd"], "n_bets": int(fin["n_bets"])}


results = {"by_league_test": {}, "by_season": {}, "leave_one_league_out": {}}

# ============ 1. Global model ============
print("[1] global XGB ...")
model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                      early_stopping_rounds=30, random_state=42)
model.fit(train[feature_cols], train["y"].values.astype(int),
          eval_set=[(val[feature_cols], val["y"].values.astype(int))], verbose=False)

# 1a. By league (test)
for div, sub in test.groupby("Div"):
    sub_odds = sub.merge(raw[["Date", "HomeTeam", "AwayTeam", "B365CH", "B365CD", "B365CA"]],
                         on=["Date", "HomeTeam", "AwayTeam"], how="left")
    results["by_league_test"][div] = eval_group(
        div, sub[feature_cols], sub["y"].values.astype(int), sub, sub_odds)
    r = results["by_league_test"][div]
    print(f"  {div}: acc={r['accuracy']:.3f} n={r['n']} roi={r['roi']*100:.2f}%")

# 1b. By season (val 2024/25 + test 2025/26)
for season, sub in feat[(feat["Date"] >= "2024-08-01")].groupby("Season"):
    sub_odds = sub.merge(raw[["Date", "HomeTeam", "AwayTeam", "B365CH", "B365CD", "B365CA"]],
                         on=["Date", "HomeTeam", "AwayTeam"], how="left")
    results["by_season"][f"{season}/{season+1}"] = eval_group(
        str(season), sub[feature_cols], sub["y"].values.astype(int), sub, sub_odds)
    r = results["by_season"][f"{season}/{season+1}"]
    print(f"  season {season}/{season+1}: acc={r['accuracy']:.3f} n={r['n']} roi={r['roi']*100:.2f}%")

# ============ 2. Leave-one-league-out ============
print("[2] Leave-one-league-out ...")
for div in ["E0", "D1", "F1", "I1", "SP1"]:
    tr = feat[(feat["Date"] < "2024-08-01") & (feat["Div"] != div)]
    va = feat[(feat["Date"] >= "2024-08-01") & (feat["Date"] < "2025-08-01") & (feat["Div"] != div)]
    te = feat[(feat["Date"] >= "2025-08-01") & (feat["Div"] == div)]
    if len(te) < 50:
        print(f"  skip {div}: test n={len(te)}")
        continue
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                      early_stopping_rounds=30, random_state=42)
    m.fit(tr[feature_cols], tr["y"].values.astype(int),
          eval_set=[(va[feature_cols], va["y"].values.astype(int))], verbose=False)
    sub_odds = te.merge(raw[["Date", "HomeTeam", "AwayTeam", "B365CH", "B365CD", "B365CA"]],
                        on=["Date", "HomeTeam", "AwayTeam"], how="left")
    y = te["y"].values.astype(int)
    proba = m.predict_proba(te[feature_cols])
    res = evaluate_predictions(y, proba)
    rets, placed = simulate_bets(y, proba, sub_odds["B365CH"].values,
                                 sub_odds["B365CD"].values, sub_odds["B365CA"].values, min_prob=0.0)
    fin = financial_metrics(rets, placed)
    results["leave_one_league_out"][div] = {
        "n": int(len(y)), "accuracy": res["accuracy"], "macro_f1": res["macro_f1"],
        "log_loss": res["log_loss"], "roi": fin["roi"], "mdd": fin["mdd"]}
    print(f"  held-out {div}: acc={res['accuracy']:.3f} n={len(y)} roi={fin['roi']*100:.2f}%")

with open(os.path.join(RES, "by_league.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("\nsaved:", os.path.join(RES, "by_league.json"))
