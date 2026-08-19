"""
Walk-forward 时序回测（跨赛季稳定性）
====================

设计：
- 测试赛季 S ∈ {2022/23, 2023/24, 2024/25, 2025/26}
- 扩展窗口：用 S 之前所有数据训练
- 滚动窗口：只用最近 2 个赛季训练（检验"遗忘旧数据"是否有益，即模型老化问题）
- 对照：市场 de-vig（无需训练）
- 指标：acc / logloss / ROI，每个测试赛季独立报告
输出：results/walkforward.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss

from evaluate import financial_metrics, simulate_bets

OUT = r"E:\论文\sci_redo\data\processed"
RES = r"E:\论文\sci_redo\results"
os.makedirs(RES, exist_ok=True)

feat = pd.read_csv(os.path.join(OUT, "all_matches_featurized.csv"), parse_dates=["Date"])
drop_cols = ["Div", "Date", "Season", "HomeTeam", "AwayTeam", "FTR", "y"]
feature_cols = [c for c in feat.columns if c not in drop_cols and feat[c].notna().sum() > 0]
# 全局 median 填充（按 train 2019-2024 拟合；walk-forward 中沿用，避免逐窗口重算引入不一致）
medians = feat[feat["Date"] < "2024-08-01"][feature_cols].median()
feat[feature_cols] = feat[feature_cols].fillna(medians)

raw = pd.concat([pd.read_csv(p) for p in glob.glob(r"E:\论文\structured_data\*.csv")], ignore_index=True)
raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])

TEST_SEASONS = {2022: "2022-08-01", 2023: "2023-08-01", 2024: "2024-08-01", 2025: "2025-08-01"}

results = {}
for year, start in TEST_SEASONS.items():
    end = "2026-08-01" if year == 2025 else TEST_SEASONS[year + 1]
    te = feat[(feat["Date"] >= start) & (feat["Date"] < end)]
    if len(te) < 100:
        print(f"skip {year}/{year+1}: n={len(te)}")
        continue
    yte = te["y"].values.astype(int)
    tmeta = te.merge(raw[["Date", "HomeTeam", "AwayTeam", "B365CH", "B365CD", "B365CA"]],
                     on=["Date", "HomeTeam", "AwayTeam"], how="left")
    inv = 1.0 / tmeta[["B365CH", "B365CD", "B365CA"]].replace(0, np.nan)
    mkt = (inv.div(inv.sum(axis=1), axis=0)).values

    entry = {"n": int(len(te))}

    # 市场基线
    acc_m = accuracy_score(yte, mkt.argmax(axis=1))
    ll_m = log_loss(yte, mkt, labels=[0, 1, 2])
    rets, placed = simulate_bets(yte, mkt, tmeta["B365CH"].values,
                                 tmeta["B365CD"].values, tmeta["B365CA"].values, min_prob=0.0)
    fin = financial_metrics(rets, placed)
    entry["market"] = {"acc": acc_m, "logloss": ll_m, "roi": fin["roi"]}

    # 扩展窗口 XGB（用训练窗口内最近 12 个月做早停验证）
    tr = feat[feat["Date"] < start]
    tr_cut = pd.Timestamp(start) - pd.DateOffset(months=12)
    tr_fit = tr[tr["Date"] < tr_cut]
    tr_val = tr[tr["Date"] >= tr_cut]
    xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                        early_stopping_rounds=30, random_state=42)
    xgb.fit(tr_fit[feature_cols], tr_fit["y"].values.astype(int),
            eval_set=[(tr_val[feature_cols], tr_val["y"].values.astype(int))], verbose=False)
    proba = xgb.predict_proba(te[feature_cols])
    acc = accuracy_score(yte, proba.argmax(axis=1))
    ll = log_loss(yte, proba, labels=[0, 1, 2])
    rets, placed = simulate_bets(yte, proba, tmeta["B365CH"].values,
                                 tmeta["B365CD"].values, tmeta["B365CA"].values, min_prob=0.0)
    fin = financial_metrics(rets, placed)
    entry["xgb_expanding"] = {"acc": acc, "logloss": ll, "roi": fin["roi"]}
    print(f"  {year}/{year+1}: market acc={acc_m:.3f} | xgb_exp acc={acc:.3f} "
          f"ll={ll:.3f} roi={fin['roi']*100:.2f}%")

    # 滚动窗口 XGB（最近 2 个赛季；同样用窗口内最近 12 个月做验证）
    cutoff2 = pd.Timestamp(start) - pd.DateOffset(years=2)
    tr2 = feat[(feat["Date"] >= cutoff2) & (feat["Date"] < start)]
    if len(tr2) > 500:
        tr2_cut = pd.Timestamp(start) - pd.DateOffset(months=12)
        tr2_fit = tr2[tr2["Date"] < tr2_cut]
        tr2_val = tr2[tr2["Date"] >= tr2_cut]
        xgb2 = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                             early_stopping_rounds=30, random_state=42)
        xgb2.fit(tr2_fit[feature_cols], tr2_fit["y"].values.astype(int),
                 eval_set=[(tr2_val[feature_cols], tr2_val["y"].values.astype(int))], verbose=False)
        proba2 = xgb2.predict_proba(te[feature_cols])
        acc2 = accuracy_score(yte, proba2.argmax(axis=1))
        ll2 = log_loss(yte, proba2, labels=[0, 1, 2])
        rets2, placed2 = simulate_bets(yte, proba2, tmeta["B365CH"].values,
                                       tmeta["B365CD"].values, tmeta["B365CA"].values, min_prob=0.0)
        fin2 = financial_metrics(rets2, placed2)
        entry["xgb_rolling2"] = {"acc": acc2, "logloss": ll2, "roi": fin2["roi"]}
        print(f"            | xgb_roll2 acc={acc2:.3f} ll={ll2:.3f} roi={fin2['roi']*100:.2f}%")

    results[f"{year}/{year+1}"] = entry

with open(os.path.join(RES, "walkforward.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("\n已保存:", os.path.join(RES, "walkforward.json"))
