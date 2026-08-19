"""
价值投注 + 开盘/收盘赔率检验 + Kelly 仓位
====================

1. 价值投注：EV = p_model * odds - 1 > threshold 才下注，扫描阈值
2. 开盘 vs 收盘：用 de-vig 开盘概率做模型、收盘赔率结算 -> 检验开盘信息含量
3. Kelly：f = (p*odds-1)/(odds-1)，封顶 10% 仓位，与等注对比
输出：results/value_betting.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from evaluate import financial_metrics

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
tmeta = test.merge(raw[["Date", "HomeTeam", "AwayTeam",
                        "B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA"]],
                   on=["Date", "HomeTeam", "AwayTeam"], how="left")


def devig(h, d, a):
    inv = 1.0 / pd.concat([h, d, a], axis=1).replace(0, np.nan)
    s = inv.sum(axis=1)
    return (inv.div(s, axis=0)).values


print("[1] 训练全局 XGB ...")
model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                      early_stopping_rounds=30, random_state=42)
model.fit(train[feature_cols], train["y"].values.astype(int),
          eval_set=[(val[feature_cols], val["y"].values.astype(int))], verbose=False)
proba = model.predict_proba(test[feature_cols])

closing = tmeta[["B365CH", "B365CD", "B365CA"]].values
opening_proba = devig(tmeta["B365H"], tmeta["B365D"], tmeta["B365A"])

results = {}

# ============ 1. 价值投注（XGB 概率 vs 收盘赔率） ============
print("\n[2] 价值投注阈值扫描（EV = p*odds - 1）...")
ev_scan = []
for thr in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]:
    rets = np.zeros(len(yte))
    placed = np.zeros(len(yte), dtype=bool)
    for i in range(len(yte)):
        p = proba[i]
        odds = closing[i]
        evs = p * odds - 1
        j = int(np.argmax(evs))
        if evs[j] > thr and np.isfinite(odds[j]) and odds[j] > 1:
            placed[i] = True
            rets[i] = (odds[j] - 1) if j == yte[i] else -1.0
    fin = financial_metrics(rets, placed)
    ev_scan.append({"threshold": thr, "roi": fin["roi"], "sharpe": fin["sharpe"],
                    "mdd": fin["mdd"], "n_bets": int(fin["n_bets"]),
                    "win_rate": fin["win_rate"],
                    "coverage": float(placed.sum() / len(yte))})
    print(f"  EV>{thr:.2f}: ROI={fin['roi']*100:.2f}% Sharpe={fin['sharpe']:.3f} "
          f"MDD={fin['mdd']*100:.1f}% n={fin['n_bets']}")
results["value_ev_scan"] = ev_scan

# ============ 2. 开盘 vs 收盘（市场内部有效性） ============
print("\n[3] 开盘 de-vig 概率 vs 收盘赔率结算 ...")
rets = np.zeros(len(yte))
placed = np.zeros(len(yte), dtype=bool)
pred_open = opening_proba.argmax(axis=1)
for i in range(len(yte)):
    odds = closing[i]
    j = pred_open[i]
    if np.isfinite(odds[j]) and odds[j] > 1:
        placed[i] = True
        rets[i] = (odds[j] - 1) if j == yte[i] else -1.0
fin = financial_metrics(rets, placed)
results["open_vs_close"] = {"roi": fin["roi"], "sharpe": fin["sharpe"], "mdd": fin["mdd"],
                            "n_bets": int(fin["n_bets"]), "win_rate": fin["win_rate"],
                            "accuracy": float((pred_open == yte).mean())}
print(f"  开盘概率@收盘结算: ROI={fin['roi']*100:.2f}% acc={(pred_open==yte).mean():.3f}")

# 反向：收盘 de-vig 概率 vs 开盘赔率结算
print("\n[4] 收盘 de-vig 概率 vs 开盘赔率结算（方向对照）...")
opening = tmeta[["B365H", "B365D", "B365A"]].values
closing_proba = devig(tmeta["B365CH"], tmeta["B365CD"], tmeta["B365CA"])
pred_close = closing_proba.argmax(axis=1)
rets = np.zeros(len(yte))
placed = np.zeros(len(yte), dtype=bool)
for i in range(len(yte)):
    odds = opening[i]
    j = pred_close[i]
    if np.isfinite(odds[j]) and odds[j] > 1:
        placed[i] = True
        rets[i] = (odds[j] - 1) if j == yte[i] else -1.0
fin = financial_metrics(rets, placed)
results["close_vs_open"] = {"roi": fin["roi"], "sharpe": fin["sharpe"], "mdd": fin["mdd"],
                            "n_bets": int(fin["n_bets"]), "win_rate": fin["win_rate"]}
print(f"  收盘概率@开盘结算: ROI={fin['roi']*100:.2f}%")

# ============ 3. Kelly 仓位 vs 等注 ============
print("\n[5] Kelly 仓位（封顶 10%）...")
bankroll = 100.0
kelly_rets = np.zeros(len(yte))
kelly_placed = np.zeros(len(yte), dtype=bool)
for i in range(len(yte)):
    p = proba[i]
    odds = closing[i]
    j = int(np.argmax(p * odds - 1))
    ev = p[j] * odds[j] - 1
    if ev <= 0 or not (np.isfinite(odds[j]) and odds[j] > 1):
        continue
    f = (p[j] * odds[j] - 1) / (odds[j] - 1)
    f = min(max(f, 0), 0.10)
    stake = f * bankroll
    kelly_placed[i] = True
    if j == yte[i]:
        ret = stake * (odds[j] - 1)
    else:
        ret = -stake
    bankroll += ret
    kelly_rets[i] = ret
fin = financial_metrics(kelly_rets, kelly_placed)
results["kelly"] = {"roi": fin["roi"], "sharpe": fin["sharpe"], "mdd": fin["mdd"],
                    "n_bets": int(fin["n_bets"]), "win_rate": fin["win_rate"],
                    "final_bankroll": float(bankroll)}
print(f"  Kelly: ROI={fin['roi']*100:.2f}% Sharpe={fin['sharpe']:.3f} "
      f"MDD={fin['mdd']*100:.1f}% 终值={bankroll:.1f} ({fin['n_bets']}注)")

# ============ 4. 校准后价值投注（验证过度自信假设） ============
print("\n[6] 概率校准（isotonic，fit on val）后重做 EV 扫描 ...")
from sklearn.isotonic import IsotonicRegression
proba_va = model.predict_proba(val[feature_cols])
yva = val["y"].values.astype(int)
calib = []
for c in range(3):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba_va[:, c], (yva == c).astype(float))
    calib.append(iso)
proba_cal = np.column_stack([calib[c].predict(proba[:, c]) for c in range(3)])
proba_cal = proba_cal / proba_cal.sum(axis=1, keepdims=True)

cal_scan = []
for thr in [0.0, 0.02, 0.05, 0.08, 0.10]:
    rets = np.zeros(len(yte))
    placed = np.zeros(len(yte), dtype=bool)
    for i in range(len(yte)):
        p = proba_cal[i]
        odds = closing[i]
        evs = p * odds - 1
        j = int(np.argmax(evs))
        if evs[j] > thr and np.isfinite(odds[j]) and odds[j] > 1:
            placed[i] = True
            rets[i] = (odds[j] - 1) if j == yte[i] else -1.0
    fin = financial_metrics(rets, placed)
    cal_scan.append({"threshold": thr, "roi": fin["roi"], "sharpe": fin["sharpe"],
                     "mdd": fin["mdd"], "n_bets": int(fin["n_bets"]),
                     "win_rate": fin["win_rate"],
                     "coverage": float(placed.sum() / len(yte))})
    print(f"  校准后 EV>{thr:.2f}: ROI={fin['roi']*100:.2f}% n={fin['n_bets']}")
results["value_ev_scan_calibrated"] = cal_scan

with open(os.path.join(RES, "value_betting.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("\n已保存:", os.path.join(RES, "value_betting.json"))
