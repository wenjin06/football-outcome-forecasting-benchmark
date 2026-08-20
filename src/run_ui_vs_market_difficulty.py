"""
诚实检验：UI 分层 vs 市场 de-vig 概率分层的独立性
================================================
问题：UI 含 (1-p_max) 项，p_max 与市场强弱高度相关。
若 UI 分层效果 ≈ 市场 de-vig max prob 分层，则"UI 是独立难度信号"不成立，
论述必须改写为"市场概率极端程度是难度信号，而价格变动/分歧不是"。

输出：results/ui_vs_market_difficulty.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from risk import fit_robust, compute_ui, risk_tiers

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
tmeta = test.merge(raw[["Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A",
                        "B365CH", "B365CD", "B365CA"]], on=["Date", "HomeTeam", "AwayTeam"], how="left")

model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                      early_stopping_rounds=30, random_state=42)
model.fit(train[feature_cols], train["y"].values.astype(int),
          eval_set=[(val[feature_cols], val["y"].values.astype(int))], verbose=False)
proba = model.predict_proba(test[feature_cols])
p_max = proba.max(axis=1)

# 市场 de-vig 概率
inv = 1.0 / tmeta[["B365CH", "B365CD", "B365CA"]].replace(0, np.nan)
s = inv.sum(axis=1)
mkt_proba = (inv.div(s, axis=0)).values
mkt_max = np.nanmax(mkt_proba, axis=1)
valid = ~np.isnan(mkt_max)

# 大热占比（初始最低赔率 <= 1.6）
init_odds = tmeta[["B365H", "B365D", "B365A"]]
fav = (init_odds.min(axis=1) <= 1.6).astype(float)

results = {}

def tier_stats(mask):
    acc = (proba[mask].argmax(1) == yte[mask]).mean()
    return {"n": int(mask.sum()), "acc": float(acc),
            "fav_share": float(fav[mask].mean()),
            "mkt_max_mean": float(np.nanmean(mkt_max[mask]))}

# 1. UI 分层（0.40/0.70 阈值，与论文一致）
vol_stats = fit_robust(train, "close_vol")
move_stats = fit_robust(train, "odds_move_H")
ui = compute_ui(proba, test, vol_stats, move_stats)
tier, _ = risk_tiers(ui, 0.30, 0.45)  # 论文实际阈值（val 上选定）
results["ui_tiers"] = {
    "low": tier_stats(tier == 0), "medium": tier_stats(tier == 1),
    "high": tier_stats(tier == 2)}
print("=== UI 分层（0.30/0.45，论文阈值）===")
for k in ["low", "medium", "high"]:
    d = results["ui_tiers"][k]
    print(f"  {k}: n={d['n']} acc={d['acc']*100:.1f}% fav={d['fav_share']*100:.0f}% mkt_max={d['mkt_max_mean']:.2f}")

# 2. 市场 de-vig max prob 分桶（5 分位）
bins = pd.qcut(pd.Series(mkt_max), 5, labels=False, duplicates="drop")
results["mkt_max_quintiles"] = []
for b in range(int(bins.max()) + 1):
    mask = (bins == b) & valid
    results["mkt_max_quintiles"].append(tier_stats(mask))
print("\n=== 市场 de-vig max prob 五分位 ===")
for b, d in enumerate(results["mkt_max_quintiles"]):
    print(f"  Q{b}: n={d['n']} acc={d['acc']*100:.1f}% fav={d['fav_share']*100:.0f}% mkt_max={d['mkt_max_mean']:.2f}")

# 3. 模型 p_max 五分位（对比：模型置信度分层）
bins_p = pd.qcut(pd.Series(p_max), 5, labels=False, duplicates="drop")
results["model_pmax_quintiles"] = []
for b in range(int(bins_p.max()) + 1):
    mask = bins_p == b
    results["model_pmax_quintiles"].append(tier_stats(mask))
print("\n=== 模型 p_max 五分位 ===")
for b, d in enumerate(results["model_pmax_quintiles"]):
    print(f"  Q{b}: n={d['n']} acc={d['acc']*100:.1f}% fav={d['fav_share']*100:.0f}% mkt_max={d['mkt_max_mean']:.2f}")

# 4. 市场 max prob 分层后，UI 是否还有增量（mkt_max 三分位 x UI 三分位交叉）
print("\n=== mkt_max 三分位 x UI 三分位交叉（增量检验）===")
try:
    mkt3 = pd.qcut(pd.Series(mkt_max), 3, labels=False, duplicates="drop")
except Exception:
    mkt3 = pd.qcut(pd.Series(mkt_max), 3, labels=False)
ui3 = pd.qcut(pd.Series(ui), 3, labels=False)
cross = []
for mb in range(3):
    for ub in range(3):
        mask = (mkt3 == mb) & (ui3 == ub) & valid
        if mask.sum() < 20:
            continue
        d = tier_stats(mask)
        cross.append({"mkt_q": int(mb), "ui_q": int(ub), **d})
for c in cross:
    print(f"  mkt_Q{c['mkt_q']} x ui_Q{c['ui_q']}: n={c['n']} acc={c['acc']*100:.1f}% fav={c['fav_share']*100:.0f}%")
results["mkt_x_ui_cross"] = cross

with open(os.path.join(RES, "ui_vs_market_difficulty.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("\n已保存:", os.path.join(RES, "ui_vs_market_difficulty.json"))
