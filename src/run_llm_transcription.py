"""
LLM transcription baseline: is the LLM output merely a direct transcription of
the market probabilities?
====================================================
Addresses the question of whether the LLM is genuinely stronger than a simple
rule that directly outputs the de-vigged market probabilities.

Analysis:
1. Transcription baseline: no LLM, directly output the de-vigged closing odds
   - Evaluated on the same 120-match subset as Table 21 (acc/logloss/brier/ece/roi),
     alongside the LLM input configurations
   - Evaluated on all 1,104 matches, alongside the LLM main row (Table 1)
2. Per-match deviation of LLM output from de-vigged market probabilities
   (all 1,104 matches):
   - Per-class MAE, per-class correlation, mean absolute deviation
   - If the deviation is ~0, the LLM acts as a calibrated interface rather than
     a prediction engine

Output: results/llm_transcription.json
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from evaluate import ece, brier_multiclass, financial_metrics

import paths
RES = paths.RES


def devig(odds):
    """odds: (n,3) raw odds -> de-vigged probabilities (n,3)"""
    inv = 1.0 / np.asarray(odds, dtype=float)
    s = inv.sum(axis=1, keepdims=True)
    return inv / s


def metrics_block(y, proba, odds=None):
    block = {
        "acc": float(accuracy_score(y, proba.argmax(axis=1))),
        "logloss": float(log_loss(y, proba, labels=[0, 1, 2])),
        "brier": float(brier_multiclass(y, proba)),
        "ece": float(ece(y, proba)),
        "n": int(len(y)),
    }
    if odds is not None:
        pred = proba.argmax(axis=1)
        rets = np.zeros(len(y))
        for i in range(len(y)):
            o = odds[i]
            j = pred[i]
            if np.isfinite(o[j]) and o[j] > 1:
                rets[i] = (o[j] - 1) if j == y[i] else -1.0
            else:
                rets[i] = np.nan
        m = ~np.isnan(rets)
        fin = financial_metrics(rets[m])
        block["roi"] = float(fin["roi"])
        block["win_rate"] = float(fin["win_rate"])
    return block


# ---- 1. 120-match subset: transcription baseline vs. LLM input ablation ----
sub = pd.read_csv(os.path.join(RES, "llm_deepseek_fmmarket_t0.3_per_match.csv"))
sub = sub.dropna(subset=["p_H"]).reset_index(drop=True)
y_sub = sub["y"].values.astype(int)
odds_sub = sub[["odds_H", "odds_D", "odds_A"]].values
mkt_sub = devig(odds_sub)
trans_120 = metrics_block(y_sub, mkt_sub, odds_sub)
print("=== transcription baseline (120-match subset, same batch as Table 21) ===")
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in trans_120.items()})

# ---- 2. All 1,104 matches: transcription baseline vs. LLM main row ----
full = pd.read_csv(os.path.join(RES, "llm_deepseek_t0.3_per_match.csv"))
full = full.dropna(subset=["p_H"]).reset_index(drop=True)
y_full = full["y"].values.astype(int)
odds_full = full[["odds_H", "odds_D", "odds_A"]].values
mkt_full = devig(odds_full)
trans_full = metrics_block(y_full, mkt_full, odds_full)
print("\n=== transcription baseline (full test set) ===")
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in trans_full.items()})

# ---- 3. Deviation of LLM output from de-vigged market probabilities (full test set) ----
p_llm = full[["p_H", "p_D", "p_A"]].values
dev = np.abs(p_llm - mkt_full)
per_class = {}
for i, cls in enumerate(["H", "D", "A"]):
    per_class[cls] = {
        "mae": float(dev[:, i].mean()),
        "corr": float(np.corrcoef(p_llm[:, i], mkt_full[:, i])[0, 1]),
    }
dev_stats = {
    "per_class": per_class,
    "mean_abs_dev": float(dev.mean()),
    "max_abs_dev": float(dev.max()),
    "pct_within_0.02": float((dev.mean(axis=1) <= 0.02).mean()),
    "pct_within_0.05": float((dev.mean(axis=1) <= 0.05).mean()),
}
print("\n=== LLM output vs market de-vig probability deviation (full test set) ===")
for k, v in per_class.items():
    print(f"  class {k}: MAE={v['mae']:.4f} corr={v['corr']:.4f}")
print(f"  mean abs dev: {dev_stats['mean_abs_dev']:.4f} "
      f"({dev_stats['pct_within_0.02']*100:.1f}% matches within 0.02)")

results = {
    "transcription_120": trans_120,
    "transcription_full": trans_full,
    "llm_vs_market_dev": dev_stats,
}
with open(os.path.join(RES, "llm_transcription.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("\nsaved:", os.path.join(RES, "llm_transcription.json"))
