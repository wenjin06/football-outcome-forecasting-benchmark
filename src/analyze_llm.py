"""
Post-processing analysis of LLM experiment results
==================================================
The risk-control section of run_llm.py used the old thresholds (0.4/0.7); here
we recompute with the correct paper thresholds (0.30/0.45) and add consistency
analysis:
1. UI-tier table (LLM, thresholds 0.30/0.45) + no-bet report
2. Consistency vs. correctness: whether consistency predicts errors (validation of the uncertainty signal)
3. Summary of main-table rows (LLM vs. market vs. XGB)
Output: results/llm_analysis.json
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

# ============ 1. UI tiers (thresholds 0.30/0.45) ============
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

# ============ 2. Consistency vs. correctness ============
correct = (proba.argmax(axis=1) == y).astype(float)
# Bin by consistency
bins = [0, 0.8, 0.9, 0.95, 1.0]
cons_table = []
for i in range(len(bins) - 1):
    mask = (cons >= bins[i]) & (cons < bins[i + 1])
    if mask.sum() < 20:
        continue
    cons_table.append({"bin": f"[{bins[i]},{bins[i+1]})", "n": int(mask.sum()),
                       "acc": float(correct[mask].mean())})
# Consistency-correctness correlation (point-biserial)
corr = np.corrcoef(cons, correct)[0, 1]
out["consistency_analysis"] = {"bins": cons_table, "point_biserial": float(corr)}
print(f"\nconsistency-correctness correlation: {corr:.3f}")
for r in cons_table:
    print(f"  cons {r['bin']}: n={r['n']} acc={r['acc']:.3f}")

# Is consistency effective as a component of UI: accuracy of high- vs. low-consistency groups
hi_c = cons >= 0.95
lo_c = cons < 0.95
if lo_c.sum() > 20:
    out["consistency_analysis"]["acc_hi"] = float(correct[hi_c].mean())
    out["consistency_analysis"]["acc_lo"] = float(correct[lo_c].mean())
    out["consistency_analysis"]["n_hi"] = int(hi_c.sum())
    out["consistency_analysis"]["n_lo"] = int(lo_c.sum())
    print(f"  high consistency (>=0.95): acc={correct[hi_c].mean():.3f} (n={hi_c.sum()})")
    print(f"  low consistency (<0.95):  acc={correct[lo_c].mean():.3f} (n={lo_c.sum()})")

# ============ 3. Main-table rows (LLM vs. market vs. XGB) ============
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
print("\nsaved:", os.path.join(RES, "llm_analysis.json"))
