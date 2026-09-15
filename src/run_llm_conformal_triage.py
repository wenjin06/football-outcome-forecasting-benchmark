"""Test whether an existing LLM run adds value inside conformal review tiers.

No API calls are made. The script aligns the frozen 1,104-match LLM output with
the chronological prefix of the refreshed complete 2025/26 test season.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import log_loss

import paths
from run_conformal_selective import normalize_probabilities


LLM_PATH = os.path.join(os.path.dirname(paths.RES), "results", "llm_deepseek_t0.3_per_match.csv")
CONFORMAL_PATH = os.path.join(paths.RES, "conformal_selective_per_match.csv")
OUT_PATH = os.path.join(paths.RES, "llm_conformal_triage.json")


def paired_test(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    a_ok, b_ok = a == y, b == y
    a_only = int((a_ok & ~b_ok).sum())
    b_only = int((~a_ok & b_ok).sum())
    discordant = a_only + b_only
    p = float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue) if discordant else 1.0
    return {"a_only_correct": a_only, "b_only_correct": b_only, "mcnemar_exact_p": p}


def main() -> None:
    conf = pd.read_csv(CONFORMAL_PATH, parse_dates=["Date"])
    llm = pd.read_csv(LLM_PATH).sort_values("idx")
    frozen_features = pd.read_csv(
        os.path.join(paths.BASE, "data", "processed", "all_matches_featurized.csv"),
        parse_dates=["Date"],
    )
    frozen_test = (
        frozen_features[frozen_features["Date"] >= "2025-08-01"]
        .sort_values("Date")
        .reset_index(drop=True)
    )
    keys = frozen_test.loc[llm["idx"].to_numpy(int), [
        "Date", "HomeTeam", "AwayTeam", "y"
    ]].reset_index(drop=True)
    keyed_llm = pd.concat([keys, llm.reset_index(drop=True).drop(columns=["y"])], axis=1)
    prefix = keyed_llm.merge(
        conf,
        on=["Date", "HomeTeam", "AwayTeam"],
        how="left",
        suffixes=("_llm_key", ""),
        validate="one_to_one",
    )
    if prefix["market_p_H"].isna().any():
        raise RuntimeError("Some frozen LLM matches are absent from the refreshed data")
    if not np.array_equal(prefix["y"].to_numpy(int), prefix["y_llm_key"].to_numpy(int)):
        raise RuntimeError("Outcome mismatch after key-based LLM alignment")

    y = prefix["y"].to_numpy(int)
    p_llm = normalize_probabilities(prefix[["p_H", "p_D", "p_A"]].to_numpy(float))
    p_market = normalize_probabilities(
        prefix[["market_p_H", "market_p_D", "market_p_A"]].to_numpy(float)
    )
    pred_llm = p_llm.argmax(axis=1)
    pred_market = p_market.argmax(axis=1)
    rows = []
    labels = np.array(["H", "D", "A"])

    for alpha in (10, 20, 30):
        masks = prefix[f"market_mondrian_mask_a{alpha:02d}"].fillna("").astype(str)
        sizes = prefix[f"market_mondrian_size_a{alpha:02d}"].to_numpy(int)
        allowed = np.stack([[label in value for label in labels] for value in masks])
        llm_restricted = np.where(allowed, p_llm, -1.0).argmax(axis=1)
        for size, action in ((1, "automated"), (2, "review"), (3, "abstain")):
            keep = sizes == size
            yy = y[keep]
            if not keep.any():
                continue
            market = pred_market[keep]
            llm_plain = pred_llm[keep]
            llm_in_set = llm_restricted[keep]
            rows.append({
                "alpha": alpha / 100,
                "action": action,
                "set_size": size,
                "n": int(keep.sum()),
                "share": float(keep.mean()),
                "market_accuracy": float((market == yy).mean()),
                "llm_accuracy": float((llm_plain == yy).mean()),
                "llm_restricted_accuracy": float((llm_in_set == yy).mean()),
                "market_log_loss": float(log_loss(yy, p_market[keep], labels=[0, 1, 2])),
                "llm_log_loss": float(log_loss(yy, p_llm[keep], labels=[0, 1, 2])),
                "market_vs_llm": paired_test(yy, market, llm_plain),
                "market_vs_llm_restricted": paired_test(yy, market, llm_in_set),
            })

    overall = {
        "n": int(len(y)),
        "market_accuracy": float((pred_market == y).mean()),
        "llm_accuracy": float((pred_llm == y).mean()),
        "market_log_loss": float(log_loss(y, p_market, labels=[0, 1, 2])),
        "llm_log_loss": float(log_loss(y, p_llm, labels=[0, 1, 2])),
        "paired": paired_test(y, pred_market, pred_llm),
    }
    result = {
        "status": "complete_existing_llm_subset",
        "note": "No new API calls; chronological 1,104-match prefix only.",
        "overall": overall,
        "rows": rows,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("overall", overall)
    for row in rows:
        if row["alpha"] == 0.10:
            print(
                f"{row['action']:10s} n={row['n']:4d} "
                f"market={row['market_accuracy']:.3f} llm={row['llm_accuracy']:.3f} "
                f"llm|set={row['llm_restricted_accuracy']:.3f} "
                f"p={row['market_vs_llm_restricted']['mcnemar_exact_p']:.3f}"
            )
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
