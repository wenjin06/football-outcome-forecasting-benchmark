"""
策略级对比：UI vs market-confidence vs SCS vs model-confidence vs random
======================================================================
回答审稿人核心质疑："为什么不直接用 market max probability 分层后 no-bet？"

统一协议（test 2025/26，XGB 概率，B365 收盘赔率，等注）：
- 每个策略有一个"风险分数"，按分数从高到低剔除（no-bet），扫描覆盖率
- 同一覆盖下比较 ROI/Sharpe/MDD/acc，画出 coverage-ROI 与 coverage-acc 曲线
- 若 UI 曲线在 market-conf-only 之上 => UI 提供增量风险分层

输出：results/policy_comparison.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from risk import fit_robust, compute_ui, compute_scs
from evaluate import financial_metrics

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
pred = proba.argmax(axis=1)
odds = tmeta[["B365CH", "B365CD", "B365CA"]].values

# 逐场收益（全下注基准）
rets = np.zeros(len(yte))
for i in range(len(yte)):
    o = odds[i]
    j = pred[i]
    if np.isfinite(o[j]) and o[j] > 1:
        rets[i] = (o[j] - 1) if j == yte[i] else -1.0
    else:
        rets[i] = np.nan
valid = ~np.isnan(rets)

# ---- 风险分数 ----
inv = 1.0 / tmeta[["B365CH", "B365CD", "B365CA"]].replace(0, np.nan)
s = inv.sum(axis=1)
mkt_proba = (inv.div(s, axis=0)).values
mkt_max = np.nanmax(mkt_proba, axis=1)          # market confidence（越大越确定）
model_max = proba.max(axis=1)                    # model confidence

vol_stats = fit_robust(train, "close_vol")
move_stats = fit_robust(train, "odds_move_H")
ui = compute_ui(proba, test, vol_stats, move_stats)   # 越大越不确定
scs = compute_scs(test, vol_stats, move_stats)        # 越大越不确定
rng = np.random.default_rng(0)
rand = rng.random(len(yte))

# 统一为"风险分数"：越大越不确定 -> 优先 no-bet
scores = {
    "UI": ui,
    "SCS": scs,
    "market_conf": -mkt_max,     # 取负：市场确定度反转
    "model_conf": -model_max,    # 模型置信度反转
    "random": rand,
}


def evaluate_at_coverage(risk, cov_frac):
    """剔除风险最高 (1-cov_frac) 比例的场次，评估剩余注单。"""
    thr = np.nanquantile(risk[valid], 1 - cov_frac)
    mask = (risk <= thr) & valid
    if mask.sum() < 20:
        return None
    r = rets[mask]
    fin = financial_metrics(r)
    acc = (pred[mask] == yte[mask]).mean()
    # bootstrap CI（ROI 与 acc）
    rngb = np.random.default_rng(42)
    n = len(r)
    roi_boot, acc_boot = [], []
    for _ in range(2000):
        idx = rngb.integers(0, n, n)
        rb = r[idx]
        roi_boot.append(rb.mean())
        acc_boot.append((pred[mask][idx] == yte[mask][idx]).mean())
    return {"coverage": float(cov_frac), "n": int(mask.sum()),
            "roi": fin["roi"], "roi_ci": [float(np.percentile(roi_boot, 2.5)),
                                             float(np.percentile(roi_boot, 97.5))],
            "acc": float(acc), "acc_ci": [float(np.percentile(acc_boot, 2.5)),
                                            float(np.percentile(acc_boot, 97.5))],
            "sharpe": fin["sharpe"], "mdd": fin["mdd"],
            "win_rate": fin["win_rate"],
            "avg_odds": float(np.nanmean(odds[mask][np.arange(mask.sum()), pred[mask]]))}


coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
results = {"coverages": coverages, "strategies": {}}
for name, sc in scores.items():
    curve = []
    for c in coverages:
        r = evaluate_at_coverage(sc, c)
        if r:
            curve.append(r)
    results["strategies"][name] = curve
    print(f"=== {name} ===")
    for r in curve:
        print(f"  cov={r['coverage']:.1f}: n={r['n']} acc={r['acc']*100:.1f}% "
              f"ROI={r['roi']*100:.2f}% [{r['roi_ci'][0]*100:.1f},{r['roi_ci'][1]*100:.1f}] "
              f"Sharpe={r['sharpe']:.3f} avg_odds={r['avg_odds']:.2f}")

with open(os.path.join(RES, "policy_comparison.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("\n已保存:", os.path.join(RES, "policy_comparison.json"))
