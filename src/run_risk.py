"""
风控实验：SCS/UI 分层 + no-bet + 止损（诚实评估）
====================
#5（no-bet 与执行注单分开报告）、#7（投注协议透明）、#8（bootstrap CI）。

策略对比（同一 XGB 概率、同一 B365 收盘赔率协议）：
  naive        : 全部下注，等注 1
  t04          : 仅 p_max>=0.4 下注
  ui_tier      : UI 低=全仓 / 中=线性缩仓 / 高=no-bet
  ui_tier_sl   : ui_tier + 止损（连亏 5 笔或回撤 10% 后降仓 0.2）
  oracle_best  : （可选参考）真实结果已知的下注——不用于论文，仅自查用

输出：results/risk_analysis.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from evaluate import evaluate_predictions, financial_metrics, simulate_bets, bootstrap_ci
from risk import (compute_scs, compute_ui, fit_robust, risk_tiers,
                  simulate_bets_with_risk)

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

raw = pd.concat([pd.read_csv(p) for p in glob.glob(r"E:\论文\structured_data\*.csv")], ignore_index=True)
raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])


def odds_for(df):
    return df.merge(raw[["Date", "HomeTeam", "AwayTeam", "B365CH", "B365CD", "B365CA"]],
                    on=["Date", "HomeTeam", "AwayTeam"], how="left")


def boot_roi_ci(rets, placed, seed=42, n_boot=2000):
    rng = np.random.default_rng(seed)
    n = len(rets)
    if placed.sum() == 0:
        return (np.nan, np.nan)
    r = rets[placed]
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(r), len(r))
        stats.append(r[idx].sum() / len(r))
    stats = np.array(stats)
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


# ============ 训练全局 XGB ============
print("[1] 训练全局 XGB ...")
model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                      early_stopping_rounds=30, random_state=42)
model.fit(train[feature_cols], train["y"].values.astype(int),
          eval_set=[(val[feature_cols], val["y"].values.astype(int))], verbose=False)

proba_tr = model.predict_proba(train[feature_cols])
proba_va = model.predict_proba(val[feature_cols])
proba_te = model.predict_proba(test[feature_cols])
yte = test["y"].values.astype(int)

# ============ SCS / UI ============
vol_stats = fit_robust(train, "close_vol")
move_stats = fit_robust(train, "odds_move_H")

scs_te = compute_scs(test, vol_stats, move_stats)
ui_te = compute_ui(proba_te, test, vol_stats, move_stats)  # consistency=1（确定性模型）
scs_va = compute_scs(val, vol_stats, move_stats)
ui_va = compute_ui(proba_va, val, vol_stats, move_stats)

tmeta = odds_for(test)
vmeta = odds_for(val)

# ============ 2. 策略对比 ============
print("\n[2] 策略对比（test）...")
UI_TL, UI_TH = 0.30, 0.45  # 按 val 分布选定的分层阈值（见 [5] 敏感性表）
strategies = {}
# naive
rets, placed = simulate_bets(yte, proba_te, tmeta["B365CH"].values,
                             tmeta["B365CD"].values, tmeta["B365CA"].values, min_prob=0.0)
fin = financial_metrics(rets, placed)
strategies["naive"] = {**fin, "roi_ci": boot_roi_ci(rets, placed)}
print(f"  naive: ROI={fin['roi']*100:.2f}% Sharpe={fin['sharpe']:.3f} "
      f"MDD={fin['mdd']*100:.1f}% ({fin['n_bets']}注)")
# t04
rets, placed = simulate_bets(yte, proba_te, tmeta["B365CH"].values,
                             tmeta["B365CD"].values, tmeta["B365CA"].values, min_prob=0.4)
fin = financial_metrics(rets, placed)
strategies["t04"] = {**fin, "roi_ci": boot_roi_ci(rets, placed)}
print(f"  t04:   ROI={fin['roi']*100:.2f}% Sharpe={fin['sharpe']:.3f} "
      f"MDD={fin['mdd']*100:.1f}% ({fin['n_bets']}注)")
# ui_tier
rets, placed, scales = simulate_bets_with_risk(
    yte, proba_te, tmeta["B365CH"].values, tmeta["B365CD"].values, tmeta["B365CA"].values,
    ui_te, t_low=UI_TL, t_hi=UI_TH, stop_loss_n=10**6, stop_loss_dd=10.0)  # 关闭止损
fin = financial_metrics(rets, placed)
strategies["ui_tier"] = {**fin, "roi_ci": boot_roi_ci(rets, placed),
                         "n_no_bet": int((risk_tiers(ui_te, UI_TL, UI_TH)[0] == 2).sum()),
                         "coverage": float(placed.sum() / len(yte))}
print(f"  ui_tier: ROI={fin['roi']*100:.2f}% Sharpe={fin['sharpe']:.3f} "
      f"MDD={fin['mdd']*100:.1f}% ({fin['n_bets']}注, no-bet={strategies['ui_tier']['n_no_bet']})")
# ui_tier + stop-loss
rets, placed, scales = simulate_bets_with_risk(
    yte, proba_te, tmeta["B365CH"].values, tmeta["B365CD"].values, tmeta["B365CA"].values,
    ui_te, t_low=UI_TL, t_hi=UI_TH, stop_loss_n=5, stop_loss_dd=0.10)
fin = financial_metrics(rets, placed)
strategies["ui_tier_sl"] = {**fin, "roi_ci": boot_roi_ci(rets, placed),
                            "coverage": float(placed.sum() / len(yte))}
print(f"  ui_tier_sl: ROI={fin['roi']*100:.2f}% Sharpe={fin['sharpe']:.3f} "
      f"MDD={fin['mdd']*100:.1f}% ({fin['n_bets']}注)")

# ============ 3. UI 风险分层表现（no-bet 单独报告） ============
print("\n[3] UI 分层表现（test, 阈值 {}/{}）...".format(UI_TL, UI_TH))
tier_te, _ = risk_tiers(ui_te, UI_TL, UI_TH)
ui_table = {}
for t, label in [(0, "low"), (1, "medium"), (2, "high(no-bet)")]:
    mask = tier_te == t
    if mask.sum() == 0:
        ui_table[label] = {"n": 0}
        continue
    acc = accuracy_score(yte[mask], proba_te[mask].argmax(axis=1))
    rets_m, placed_m = simulate_bets(yte[mask], proba_te[mask],
                                     tmeta["B365CH"].values[mask], tmeta["B365CD"].values[mask],
                                     tmeta["B365CA"].values[mask], min_prob=0.0)
    fin = financial_metrics(rets_m, placed_m)
    ui_table[label] = {"n": int(mask.sum()), "accuracy": acc, **fin}
    print(f"  {label}: n={mask.sum()} acc={acc:.3f} ROI={fin['roi']*100:.2f}% "
          f"MDD={fin['mdd']*100:.1f}% ({fin['n_bets']}注)")

# ============ 4. SCS 分层表现 ============
print("\n[4] SCS 分层表现（test, 阈值 1.4/1.8）...")
scs_table = {}
for label, mask in [("low(SCS<=1.4)", scs_te <= 1.4), ("mid(1.4<SCS<=1.8)", (scs_te > 1.4) & (scs_te <= 1.8)),
                    ("high(SCS>1.8)", scs_te > 1.8)]:
    if mask.sum() == 0:
        scs_table[label] = {"n": 0}
        continue
    acc = accuracy_score(yte[mask], proba_te[mask].argmax(axis=1))
    rets_m, placed_m = simulate_bets(yte[mask], proba_te[mask],
                                     tmeta["B365CH"].values[mask], tmeta["B365CD"].values[mask],
                                     tmeta["B365CA"].values[mask], min_prob=0.0)
    fin = financial_metrics(rets_m, placed_m)
    scs_table[label] = {"n": int(mask.sum()), "accuracy": acc, **fin}
    print(f"  {label}: n={mask.sum()} acc={acc:.3f} ROI={fin['roi']*100:.2f}%")

# ============ 5. 阈值敏感性（val 上网格，test 报告固定阈值） ============
print("\n[5] 阈值敏感性（val 网格搜索）...")
sensitivity = {"scs_threshold": {}, "ui_threshold": {}}
# SCS 阈值：路由到规则引擎(市场 de-vig argmax) vs XGB，组合准确率+logloss
mkt_proba_va = vmeta[["mkt_prob_H", "mkt_prob_D", "mkt_prob_A"]].values
yva = val["y"].values.astype(int)
from sklearn.metrics import log_loss as sk_logloss
for t in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
    rule_mask = scs_va <= t
    pred = np.where(rule_mask, mkt_proba_va.argmax(axis=1), proba_va.argmax(axis=1))
    acc = accuracy_score(yva, pred)
    # 混合概率（规则侧用市场概率，LLM/XGB 侧用 XGB 概率）
    proba_mix = np.where(rule_mask[:, None], mkt_proba_va, proba_va)
    ll = sk_logloss(yva, proba_mix, labels=[0, 1, 2])
    sensitivity["scs_threshold"][str(t)] = {"val_acc": acc, "val_logloss": ll,
                                             "rule_frac": float(rule_mask.mean())}
    print(f"  SCS t={t}: val_acc={acc:.4f} val_logloss={ll:.4f} rule_frac={rule_mask.mean():.2f}")
# UI 阈值：val 上 (t_low, t_hi) 网格 -> no-bet 覆盖率与 ROI
vmeta2 = vmeta
for tl, th in [(0.25, 0.45), (0.30, 0.45), (0.30, 0.50), (0.35, 0.50), (0.40, 0.55)]:
    rets, placed, _ = simulate_bets_with_risk(
        yva, proba_va, vmeta2["B365CH"].values, vmeta2["B365CD"].values, vmeta2["B365CA"].values,
        ui_va, t_low=tl, t_hi=th, stop_loss_n=10**6, stop_loss_dd=10.0)
    fin = financial_metrics(rets, placed)
    n_nb = int((risk_tiers(ui_va, tl, th)[0] == 2).sum())
    sensitivity["ui_threshold"][f"{tl}/{th}"] = {"roi": fin["roi"], "mdd": fin["mdd"],
                                                 "n_bets": int(fin["n_bets"]), "n_no_bet": n_nb,
                                                 "coverage": float(placed.sum() / len(yva))}
    print(f"  UI t_low={tl} t_hi={th}: ROI={fin['roi']*100:.2f}% "
          f"coverage={placed.sum()/len(yva):.2f} no-bet={n_nb} ({fin['n_bets']}注)")

# ============ 保存 ============
results = {
    "strategies": {k: {kk: vv for kk, vv in v.items()} for k, v in strategies.items()},
    "ui_tiers": ui_table, "scs_tiers": scs_table,
    "sensitivity": sensitivity,
    "ui_thresholds_used": {"t_low": UI_TL, "t_hi": UI_TH},
    "scs_threshold_used": 0.4,
}
with open(os.path.join(RES, "risk_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("\n已保存:", os.path.join(RES, "risk_analysis.json"))
