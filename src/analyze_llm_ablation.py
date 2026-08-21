"""
LLM input ablation analysis: market vs. stats vs. market_stats vs. full (same first 120 matches)
================================================================================================
Determines whether the LLM's market-level calibration stems from the closing odds
included in the prompt or from genuine reasoning over non-market information.

Output: results/llm_ablation.json
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from evaluate import brier_multiclass, ece, financial_metrics, simulate_bets

import paths
RES = paths.RES


def load(name, n=120):
    df = pd.read_csv(os.path.join(RES, name))
    df = df.dropna(subset=["p_H"]).head(n).reset_index(drop=True)
    proba = df[["p_H", "p_D", "p_A"]].values
    y = df["y"].values.astype(int)
    odds = df[["odds_H", "odds_D", "odds_A"]].values
    return y, proba, odds


files = {
    "market_only": "llm_deepseek_fmmarket_t0.3_per_match.csv",
    "stats_only": "llm_deepseek_fmstats_t0.3_per_match.csv",
    "market_stats": "llm_deepseek_fmmarket_stats_t0.3_per_match.csv",
    "full": "llm_deepseek_t0.3_per_match.csv",
}

out = {}
print("=== LLM input ablation (same first 120 matches) ===")
for label, fn in files.items():
    p = os.path.join(RES, fn)
    if not os.path.exists(p):
        print(f"  {label}: MISSING ({fn})")
        continue
    y, proba, odds = load(fn)
    res = {
        "acc": float(accuracy_score(y, proba.argmax(axis=1))),
        "logloss": float(log_loss(y, proba, labels=[0, 1, 2])),
        "brier": float(brier_multiclass(y, proba)),
        "ece": float(ece(y, proba)),
        "n": int(len(y)),
    }
    rets, placed = simulate_bets(y, proba, odds[:, 0], odds[:, 1], odds[:, 2], min_prob=0.0)
    fin = financial_metrics(rets, placed)
    res["roi"] = fin["roi"]
    res["win_rate"] = fin["win_rate"]
    out[label] = res
    print(f"  {label}: acc={res['acc']:.3f} logloss={res['logloss']:.4f} "
          f"ece={res['ece']:.4f} roi={res['roi']*100:.2f}%")

with open(os.path.join(RES, "llm_ablation.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=float)
print("\nsaved:", os.path.join(RES, "llm_ablation.json"))
