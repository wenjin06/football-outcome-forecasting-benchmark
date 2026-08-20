"""
平局深度分析：为什么模型几乎不预测平局？
========================================
分析 test 2025/26：
1. 模型 p_draw 分布（分桶） vs 每桶实际平局率
2. 市场 de-vig p_draw 分布 vs 模型 p_draw
3. 按类校准（H/D/A 各自的 ECE）
4. 多分类 log-loss 按类分解
5. 类先验（train/val/test）

输出：results/draw_deep.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss

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
yte = test["y"].values.astype(int)

raw = pd.concat([pd.read_csv(p) for p in glob.glob(r"E:\论文\structured_data\*.csv")], ignore_index=True)
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

# 类先验
for name, part in [("train", train), ("val", val), ("test", test)]:
    y = part["y"].values.astype(int)
    results[f"prior_{name}"] = {"H": float((y == 0).mean()), "D": float((y == 1).mean()),
                                "A": float((y == 2).mean())}
print("priors:", {k: {kk: round(v, 3) for kk, v in vv.items()} for k, vv in results.items() if k.startswith("prior")})

# 1. 模型 p_draw 分布
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
print("\n模型 p_draw 分桶:")
for r in rows:
    print(f"  {r['bin']}: n={r['n']} mean_p={r['mean_p']:.3f} actual={r['actual_draw_rate']*100:.1f}%")

# 2. 市场 p_draw 分布
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
print("\n市场 de-vig p_draw 分桶:")
for r in rows2:
    print(f"  {r['bin']}: n={r['n']} mean_p={r['mean_p']:.3f} actual={r['actual_draw_rate']*100:.1f}%")

# 3. 按类校准 ECE（每类独立）
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
print("\n按类 ECE:", {k: round(v, 4) for k, v in results["class_ece"].items()})

# 4. 多分类 log-loss 按类分解：每类的 -log p(真实类) 均值
pm = proba[maskv][np.arange(maskv.sum()), yte[maskv]]
ll_by_class = {}
for c in range(3):
    m = (yte[maskv] == c)
    if m.sum() > 0:
        ll_by_class[str(c)] = float(-np.log(pm[m]).mean())
results["logloss_by_class"] = ll_by_class
print("\n按类 log-loss:", {k: round(v, 4) for k, v in ll_by_class.items()})

# 5. 相关：p_draw 与实际平局
results["corr_pdraw_draw"] = float(np.corrcoef(pd_model[maskv], (yte[maskv] == 1).astype(float))[0, 1])
results["corr_mktpdraw_draw"] = float(np.corrcoef(pd_mkt[maskv], (yte[maskv] == 1).astype(float))[0, 1])
print("\ncorr(model p_draw, draw):", round(results["corr_pdraw_draw"], 3))
print("corr(market p_draw, draw):", round(results["corr_mktpdraw_draw"], 3))

with open(os.path.join(RES, "draw_deep.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("\n已保存:", os.path.join(RES, "draw_deep.json"))
