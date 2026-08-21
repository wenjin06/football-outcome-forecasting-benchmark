"""
Cost-sensitive draw model (sample-weighted training, beyond threshold scanning)
====================
Extension of recommended enhancement #1: instead of post-hoc decision-threshold
tuning, the draw class is weighted during training.
- XGB sample weights: draw weight w in {1.5, 2.0, 2.5, 3.0} (home/away = 1)
- Compared: macro F1 / draw recall / acc / logloss
Output: results/draw_cost_sensitive.json
"""
import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, log_loss

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

results = {}
for w_draw in [1.0, 1.5, 2.0, 2.5, 3.0]:
    wt = np.where(train["y"].values == 1, w_draw, 1.0)
    m = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                      early_stopping_rounds=30, random_state=42)
    m.fit(train[feature_cols], train["y"].values.astype(int),
          sample_weight=wt,
          eval_set=[(val[feature_cols], val["y"].values.astype(int))], verbose=False)
    proba = m.predict_proba(test[feature_cols])
    pred = proba.argmax(axis=1)
    acc = accuracy_score(yte, pred)
    mf1 = f1_score(yte, pred, average="macro")
    p, r, f, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2])
    ll = log_loss(yte, proba, labels=[0, 1, 2])
    results[str(w_draw)] = {"acc": acc, "macro_f1": mf1, "logloss": ll,
                            "draw_precision": p[1], "draw_recall": r[1], "draw_f1": f[1],
                            "draw_pred_rate": float((pred == 1).mean())}
    print(f"  w_draw={w_draw}: acc={acc:.3f} macroF1={mf1:.3f} drawR={r[1]:.3f} "
          f"drawP={p[1]:.3f} ll={ll:.3f}")

with open(os.path.join(RES, "draw_cost_sensitive.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("saved:", os.path.join(RES, "draw_cost_sensitive.json"))
