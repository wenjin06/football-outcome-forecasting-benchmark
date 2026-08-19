"""
消融实验（非 LLM 部分，立即可以跑出真数字）
====================
在无泄漏管道产物上，对 XGBoost 做特征组消融：
- full          : 全部赛前特征
- no_odds       : 去掉全部市场信号（隐含概率/赔率变动/离散度）
- no_referee    : 去掉裁判历史特征
- no_rank       : 去掉联赛排名（保留滚动状态/赛季积分/市场）
- structured_only: 只有球队状态+排名（无市场、无裁判）
- no_form       : 去掉滚动状态与赛季积分（保留市场/裁判；排名同时去掉，因其由积分定义）

同一超参、同一 val 早停、同一 test 评估。结果存 results/ablations_summary.json
"""
import os
import json
import glob
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from evaluate import evaluate_predictions, financial_metrics, simulate_bets

OUT = r"E:\论文\sci_redo\data\processed"
RES = r"E:\论文\sci_redo\results"
os.makedirs(RES, exist_ok=True)

feat = pd.read_csv(os.path.join(OUT, "all_matches_featurized.csv"), parse_dates=["Date"])
drop_cols = ["Div", "Date", "Season", "HomeTeam", "AwayTeam", "FTR", "y"]
all_cols = [c for c in feat.columns if c not in drop_cols and feat[c].notna().sum() > 0]

ODDS_COLS = [c for c in all_cols if c.startswith("mkt_prob") or c.startswith("odds_move")
             or c in ("close_vol", "close_avg_h", "open_avg_h", "close_max_h")]
REF_COLS = [c for c in all_cols if c.startswith("ref_")]
RANK_COLS = [c for c in all_cols if c.endswith("_rank")]
ROLL_COLS = [c for c in all_cols if c.startswith(("H_roll", "A_roll"))]
SEASON_COLS = [c for c in all_cols if c.startswith(("H_season", "A_season"))]
XG_COLS = [c for c in all_cols if c.startswith(("H_x", "A_x")) or c == "xg_diff_roll"]

GROUPS = {
    "full": all_cols,
    "no_odds": [c for c in all_cols if c not in ODDS_COLS],
    "no_referee": [c for c in all_cols if c not in REF_COLS],
    "no_rank": [c for c in all_cols if c not in RANK_COLS],
    "no_xg": [c for c in all_cols if c not in XG_COLS],
    "structured_only": [c for c in all_cols if c not in ODDS_COLS + REF_COLS],
    "no_form": [c for c in all_cols if c not in ROLL_COLS + SEASON_COLS + RANK_COLS],
}

train = feat[feat["Date"] < "2024-08-01"]
val = feat[(feat["Date"] >= "2024-08-01") & (feat["Date"] < "2025-08-01")]
test = feat[feat["Date"] >= "2025-08-01"]

# 测试集赔率（财务模拟用）
raw = pd.concat([pd.read_csv(p) for p in glob.glob(r"E:\论文\structured_data\*.csv")], ignore_index=True)
raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
tmeta = test.merge(raw[["Date", "HomeTeam", "AwayTeam", "B365CH", "B365CD", "B365CA"]],
                   on=["Date", "HomeTeam", "AwayTeam"], how="left")

results = {}
for name, cols in GROUPS.items():
    print(f"\n===== {name} ({len(cols)} features) =====")
    med = train[cols].median()
    Xtr = train[cols].fillna(med)
    Xva = val[cols].fillna(med)
    Xte = test[cols].fillna(med)
    ytr = train["y"].values.astype(int)
    yva = val["y"].values.astype(int)
    yte = test["y"].values.astype(int)

    xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                        early_stopping_rounds=30, random_state=42)
    xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    proba = xgb.predict_proba(Xte)

    res = evaluate_predictions(yte, proba)
    rets, placed = simulate_bets(yte, proba, tmeta["B365CH"].values, tmeta["B365CD"].values,
                                 tmeta["B365CA"].values, min_prob=0.0)
    fin = financial_metrics(rets, placed)
    rets2, placed2 = simulate_bets(yte, proba, tmeta["B365CH"].values, tmeta["B365CD"].values,
                                   tmeta["B365CA"].values, min_prob=0.4)
    fin2 = financial_metrics(rets2, placed2)

    # bootstrap acc CI
    from sklearn.metrics import accuracy_score
    rng = np.random.default_rng(42)
    n = len(yte)
    acc_boot = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        acc_boot.append(accuracy_score(yte[idx], proba[idx].argmax(axis=1)))
    acc_ci = (np.percentile(acc_boot, 2.5), np.percentile(acc_boot, 97.5))

    results[name] = {
        "n_features": len(cols),
        "accuracy": res["accuracy"], "acc_ci": acc_ci,
        "macro_f1": res["macro_f1"], "log_loss": res["log_loss"],
        "brier": res["brier"], "ece": res["ece"],
        "roi_all": fin["roi"], "mdd_all": fin["mdd"], "winrate_all": fin["win_rate"],
        "roi_t04": fin2["roi"], "mdd_t04": fin2["mdd"], "winrate_t04": fin2["win_rate"],
        "n_bets_t04": int(fin2["n_bets"]),
    }
    print(f"  acc={res['accuracy']:.4f} [{acc_ci[0]:.4f},{acc_ci[1]:.4f}] "
          f"logloss={res['log_loss']:.4f} ece={res['ece']:.4f} "
          f"roi_all={fin['roi']*100:.2f}% roi_t04={fin2['roi']*100:.2f}%")

with open(os.path.join(RES, "ablations_summary.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("\n已保存:", os.path.join(RES, "ablations_summary.json"))
