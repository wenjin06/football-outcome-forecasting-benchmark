"""
Feature importance (permutation importance, interpretability evidence)
====================
Quantitative evidence that market odds dominate the predictions (cross-check of
the ablation conclusions).
- Permutation importance: shuffle one feature at a time and measure the test
  log-loss increase; n_repeats=5
- Output: results/feature_importance.json (top 20 + group summary)
"""
import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from sklearn.inspection import permutation_importance

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

print("[1] training global XGB ...")
model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                      early_stopping_rounds=30, random_state=42)
model.fit(train[feature_cols], train["y"].values.astype(int),
          eval_set=[(val[feature_cols], val["y"].values.astype(int))], verbose=False)

Xte = test[feature_cols].values
yte = test["y"].values.astype(int)

print("[2] permutation importance (test, n_repeats=5)...")
pi = permutation_importance(model, Xte, yte, scoring="neg_log_loss",
                            n_repeats=5, random_state=42, n_jobs=-1)

imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": pi.importances_mean,
    "std": pi.importances_std,
}).sort_values("importance", ascending=False)

# Group summary
def group_of(f):
    if f.startswith("mkt_prob") or f.startswith("odds_move") or f in ("close_vol", "close_avg_h", "open_avg_h", "close_max_h"):
        return "market"
    if f.startswith(("H_x", "A_x", "H_deep", "A_deep", "H_xpts", "A_xpts", "H_ppda", "A_ppda")) or f == "xg_diff_roll":
        return "xg"
    if f.startswith("ref_"):
        return "referee"
    if f.endswith("_rank"):
        return "rank"
    if f.startswith(("H_roll", "A_roll")):
        return "form_rolling"
    return "season_cum"

imp["group"] = imp["feature"].apply(group_of)
group_sum = imp.groupby("group")["importance"].sum().sort_values(ascending=False)

out = {
    "top20": imp.head(20).to_dict("records"),
    "group_importance": group_sum.to_dict(),
    "baseline_logloss": float(log_loss(yte, model.predict_proba(Xte), labels=[0, 1, 2])),
}
with open(os.path.join(RES, "feature_importance.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=float)

print("top 10:")
print(imp.head(10).to_string(index=False))
print("\ngroup importance (total logloss increase):")
print(group_sum.round(4).to_string())
print("\nsaved:", os.path.join(RES, "feature_importance.json"))
