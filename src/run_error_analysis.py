"""
错误分析：失败预测与赔率异动 / 市场分歧的关联
============================================
EXPERIMENT_PLAN 第 10 节待建项：
  错误分析（失败与赔率异动/分歧的关联）→ run_error_analysis.py

回答的问题（全部在 test 2025/26 上半季 1104 场上，与主实验同协议）：
1. 模型-市场分歧：XGB argmax 与市场 de-vig argmax 不一致的场次，谁更常对？
   （直接证据：分歧大时模型能否赢过市场 —— RQ1 市场有效性）
2. 赔率异动（收盘-初盘变动幅度）分桶下的模型错误率：异动大的场次是否更难预测？
3. 市场分歧（多家博彩报价离散度）分桶下的错误率：各家分歧大的场次是否更难预测？
4. 高置信错误：模型高置信（max prob>=0.6）但预测错误的占比 vs 市场同口径
   （过度自信证据，呼应 value_betting 的结论）
5. 平局专项：被误分类的平局场次 vs 正确分类的平局场次，在异动/分歧/置信上是否有差异

输出：results/error_analysis.json
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

# ---------------- 数据加载（与 run_value_betting.py 完全同协议） ----------------
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


# ---------------- 模型（与主实验同参数） ----------------
print("[1] 训练全局 XGB ...")
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

# ---------------- 1. 模型-市场分歧：分歧场次谁更常对 ----------------
print("\n[2] 模型 vs 市场：分歧场次 ...")
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
print(f"  总体 acc: 模型 {acc(err[valid]):.3f} / 市场 {acc(mkt_err[valid]):.3f}")
print(f"  分歧 {int(disagree.sum())} 场: 模型 {acc(err[disagree]):.3f} / 市场 {acc(mkt_err[disagree]):.3f}")
print(f"  一致 {int(agree.sum())} 场: 模型 {acc(err[agree]):.3f} / 市场 {acc(mkt_err[agree]):.3f}")

# ---------------- 2. 赔率异动分桶（收盘-初盘，B365） ----------------
print("\n[3] 赔率异动分桶 ...")
move_h = tmeta["B365CH"] - tmeta["B365H"]
move_d = tmeta["B365CD"] - tmeta["B365D"]
move_a = tmeta["B365CA"] - tmeta["B365A"]
move = (move_h.abs() + move_d.abs() + move_a.abs()) / 3.0
# 混杂诊断：大热占比（初始最低赔率<=1.6）与市场 max de-vig 概率
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
    print(f"  异动桶{r['bin']} {r['move_range']}: n={r['n']} 错误率={r['error_rate']*100:.1f}% "
          f"大热占比={r['fav_share']*100:.0f}% 市场maxP={r['market_max_prob']:.2f}")

# ---------------- 3. 市场分歧分桶（多家报价离散度，收盘） ----------------
print("\n[4] 市场分歧分桶 ...")
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
    print(f"  分歧桶{r['bin']} {r['disp_range']}: n={r['n']} 错误率={r['error_rate']*100:.1f}% "
          f"大热占比={r['fav_share']*100:.0f}% 市场maxP={r['market_max_prob']:.2f}")

# ---------------- 4. 高置信错误（过度自信证据） ----------------
print("\n[5] 高置信错误 ...")
conf = proba.max(axis=1)
mkt_conf = market_proba.max(axis=1)
for thr in [0.5, 0.6, 0.7]:
    hm = conf >= thr
    mm = (mkt_conf >= thr) & valid
    results[f"high_conf_{int(thr*100)}"] = {
        "model_n": int(hm.sum()), "model_error_rate": float(err[hm].mean()),
        "market_n": int(mm.sum()), "market_error_rate": float(mkt_err[mm].mean()),
    }
    print(f"  conf>={thr}: 模型 n={int(hm.sum())} 错误率={err[hm].mean()*100:.1f}% "
          f"| 市场 n={int(mm.sum())} 错误率={mkt_err[mm].mean()*100:.1f}%")

# ---------------- 5. 平局专项错误 ----------------
print("\n[6] 平局误分类分析 ...")
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
    print(f"  平局真实 {int(draw_true.sum())} 场: 正确 {int(draw_correct.sum())} / 误分 {int(draw_wrong.sum())}")
    print(f"  误分: 异动 {move[draw_wrong].mean():.3f} 分歧 {disp[draw_wrong].mean():.4f} 置信 {conf[draw_wrong].mean():.3f}")
    print(f"  正确: 异动 {move[draw_correct].mean():.3f} 分歧 {disp[draw_correct].mean():.4f} 置信 {conf[draw_correct].mean():.3f}")

# ---------------- 保存 ----------------
with open(os.path.join(RES, "error_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("\n已保存:", os.path.join(RES, "error_analysis.json"))
