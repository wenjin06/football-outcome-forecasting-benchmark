"""
LLM 实验结果后处理分析
====================
run_llm.py 的风控段用了旧阈值（0.4/0.7），这里用论文正确阈值（0.30/0.45）重算，
并补充一致性分析：
1. UI 分层表（LLM，阈值 0.30/0.45）+ no-bet 报告
2. 一致性 vs 正确性：一致性是否预测错误（不确定性信号验证）
3. LLM vs 市场 vs XGB 主表行汇总
输出：results/llm_analysis.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from evaluate import financial_metrics, simulate_bets
from risk import risk_tiers

RES = r"E:\论文\sci_redo\results"

pm = pd.read_csv(os.path.join(RES, "llm_deepseek_t0.3_per_match.csv"))
pm = pm.dropna(subset=["p_H"]).reset_index(drop=True)
proba = pm[["p_H", "p_D", "p_A"]].values
y = pm["y"].values.astype(int)
odds = pm[["odds_H", "odds_D", "odds_A"]].values
ui = pm["ui"].values
cons = pm["consistency"].values
n = len(pm)

out = {"n_matches": n}

# ============ 1. UI 分层（阈值 0.30/0.45） ============
tier, _ = risk_tiers(ui, 0.30, 0.45)
tiers = {}
for t, label in [(0, "low"), (1, "medium"), (2, "high(no-bet)")]:
    mask = tier == t
    if mask.sum() == 0:
        tiers[label] = {"n": 0}
        continue
    acc = accuracy_score(y[mask], proba[mask].argmax(axis=1))
    rets = np.zeros(mask.sum())
    placed = np.zeros(mask.sum(), dtype=bool)
    sub_y, sub_p, sub_odds = y[mask], proba[mask], odds[mask]
    for i in range(mask.sum()):
        j = int(sub_p[i].argmax())
        o = sub_odds[i, j]
        if np.isfinite(o) and o > 1:
            placed[i] = True
            rets[i] = (o - 1) if j == sub_y[i] else -1.0
    fin = financial_metrics(rets, placed)
    tiers[label] = {"n": int(mask.sum()), "accuracy": acc,
                    "roi": fin["roi"], "sharpe": fin["sharpe"], "mdd": fin["mdd"]}
    print(f"  UI {label}: n={mask.sum()} acc={acc:.3f} ROI={fin['roi']*100:.2f}%")
out["ui_tiers"] = tiers

# ============ 2. 一致性 vs 正确性 ============
correct = (proba.argmax(axis=1) == y).astype(float)
# 按一致性分箱
bins = [0, 0.8, 0.9, 0.95, 1.0]
cons_table = []
for i in range(len(bins) - 1):
    mask = (cons >= bins[i]) & (cons < bins[i + 1])
    if mask.sum() < 20:
        continue
    cons_table.append({"bin": f"[{bins[i]},{bins[i+1]})", "n": int(mask.sum()),
                       "acc": float(correct[mask].mean())})
# 一致性-正确性相关系数（点二列相关）
corr = np.corrcoef(cons, correct)[0, 1]
out["consistency_analysis"] = {"bins": cons_table, "point_biserial": float(corr)}
print(f"\n一致性-正确性相关: {corr:.3f}")
for r in cons_table:
    print(f"  cons {r['bin']}: n={r['n']} acc={r['acc']:.3f}")

# 一致性作为 UI 的组成部分是否有效：高一致性组准确率 vs 低一致性组
hi_c = cons >= 0.95
lo_c = cons < 0.95
if lo_c.sum() > 20:
    out["consistency_analysis"]["acc_hi"] = float(correct[hi_c].mean())
    out["consistency_analysis"]["acc_lo"] = float(correct[lo_c].mean())
    out["consistency_analysis"]["n_hi"] = int(hi_c.sum())
    out["consistency_analysis"]["n_lo"] = int(lo_c.sum())
    print(f"  高一致性(>=0.95): acc={correct[hi_c].mean():.3f} (n={hi_c.sum()})")
    print(f"  低一致性(<0.95):  acc={correct[lo_c].mean():.3f} (n={lo_c.sum()})")

# ============ 3. 主表行（LLM vs 市场 vs XGB） ============
from sklearn.metrics import log_loss as sk_ll
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import brier_multiclass, ece

def row(name, p, y_, odds_, min_prob=0.0):
    acc = accuracy_score(y_, p.argmax(axis=1))
    ll = sk_ll(y_, p, labels=[0, 1, 2])
    b = brier_multiclass(y_, p)
    ec = ece(y_, p)
    rets = np.zeros(len(y_))
    placed = np.zeros(len(y_), dtype=bool)
    for i in range(len(y_)):
        j = int(p[i].argmax())
        o = odds_[i, j]
        if p[i].max() < min_prob or not (np.isfinite(o) and o > 1):
            continue
        placed[i] = True
        rets[i] = (o - 1) if j == y_[i] else -1.0
    fin = financial_metrics(rets, placed)
    return {"acc": acc, "logloss": ll, "brier": b, "ece": ec,
            "roi": fin["roi"], "sharpe": fin["sharpe"], "mdd": fin["mdd"],
            "win_rate": fin["win_rate"], "n_bets": int(fin["n_bets"])}

out["main_row_llm"] = row("LLM", proba, y, odds)

with open(os.path.join(RES, "llm_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=float)
print("\n已保存:", os.path.join(RES, "llm_analysis.json"))
