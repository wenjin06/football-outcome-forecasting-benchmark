"""
====================
1. 混淆矩阵（全局 XGB vs 市场基线）
2. 经验平局率 vs 模型预测平局率
3. 平局类 P/R/F1 + 类别校准（分箱 acc vs conf）
4. 决策阈值扫描：调平局阈值能否提升 macro-F1 / 平局 recall
输出：results/draw_analysis.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from evaluate import ece, reliability_table

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
tmeta = test.merge(raw[["Date", "HomeTeam", "AwayTeam", "B365CH", "B365CD", "B365CA"]],
                   on=["Date", "HomeTeam", "AwayTeam"], how="left")

# 全局 XGB
print("[1] 训练全局 XGB ...")
model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                      early_stopping_rounds=30, random_state=42)
model.fit(train[feature_cols], train["y"].values.astype(int),
          eval_set=[(val[feature_cols], val["y"].values.astype(int))], verbose=False)
proba_xgb = model.predict_proba(test[feature_cols])

# 市场基线概率
inv = 1.0 / tmeta[["B365CH", "B365CD", "B365CA"]].replace(0, np.nan)
s = inv.sum(axis=1)
proba_mkt = (inv.div(s, axis=0)).values

results = {"empirical_draw_rate": float((yte == 1).mean()),
           "draw_rate_by_league": test.groupby("Div")["y"].apply(lambda x: float((x == 1).mean())).to_dict()}

for name, proba in [("xgb", proba_xgb), ("market", proba_mkt)]:
    pred = proba.argmax(axis=1)
    cm = confusion_matrix(yte, pred, labels=[0, 1, 2])
    p, r, f, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2])
    # 类别校准：按各类别预测概率分箱
    class_ece = {}
    for cls in [0, 1, 2]:
        conf = proba[:, cls]
        acc = (yte == cls).astype(float)
        bins = np.linspace(0, 1, 11)
        e = 0.0
        for i in range(10):
            mask = (conf > bins[i]) & (conf <= bins[i + 1])
            if mask.sum() == 0:
                continue
            e += (mask.sum() / len(yte)) * abs(acc[mask].mean() - conf[mask].mean())
        class_ece[str(cls)] = float(e)
    results[name] = {
        "confusion_matrix": cm.tolist(),
        "pred_draw_rate": float((pred == 1).mean()),
        "class_precision": p.tolist(), "class_recall": r.tolist(), "class_f1": f.tolist(),
        "class_ece": class_ece,
    }
    print(f"  {name}: 预测平局率={results[name]['pred_draw_rate']:.3f} "
          f"(经验 {results['empirical_draw_rate']:.3f})")
    print(f"    P={p.round(3)} R={r.round(3)} F1={f.round(3)} classECE={[round(v,3) for v in class_ece.values()]}")

# 阈值扫描：提高平局决策阈值（重加权），看 macro-F1 与平局 recall
print("\n[2] 平局阈值扫描 (XGB) ...")
scan = []
for w_draw in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    weighted = proba_xgb.copy()
    weighted[:, 1] *= w_draw
    pred = weighted.argmax(axis=1)
    from sklearn.metrics import f1_score
    mf1 = f1_score(yte, pred, average="macro")
    p, r, f, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2])
    acc = (pred == yte).mean()
    scan.append({"w_draw": w_draw, "macro_f1": mf1, "draw_recall": r[1],
                 "draw_precision": p[1], "draw_f1": f[1], "accuracy": acc})
    print(f"  w_draw={w_draw}: macroF1={mf1:.3f} drawR={r[1]:.3f} drawP={p[1]:.3f} acc={acc:.3f}")
results["threshold_scan"] = scan

with open(os.path.join(RES, "draw_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("\n已保存:", os.path.join(RES, "draw_analysis.json"))
