"""Multi-seed robustness and dependence-aware paired tests on complete 2025/26."""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

import paths
from run_conformal_selective import normalize_probabilities


SEEDS = (7, 42, 101, 2026, 31415)
OUT_PATH = os.path.join(paths.RES, "multiseed_paired.json")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "accuracy": float((p.argmax(axis=1) == y).mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
    }


def paired_summary(y: np.ndarray, p_model: np.ndarray, p_market: np.ndarray,
                   block: np.ndarray, seed: int = 42) -> dict:
    pred_m = p_model.argmax(axis=1)
    pred_b = p_market.argmax(axis=1)
    m_ok, b_ok = pred_m == y, pred_b == y
    model_only = int((m_ok & ~b_ok).sum())
    market_only = int((~m_ok & b_ok).sum())
    discordant = model_only + market_only
    exact_p = float(binomtest(min(model_only, market_only), discordant, 0.5).pvalue) if discordant else 1.0
    ll_diff_i = -np.log(p_model[np.arange(len(y)), y]) + np.log(
        p_market[np.arange(len(y)), y]
    )
    unique_blocks = np.unique(block)
    members = {b: np.flatnonzero(block == b) for b in unique_blocks}
    rng = np.random.default_rng(seed)
    acc_diffs, ll_diffs = [], []
    for _ in range(5000):
        sampled = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([members[b] for b in sampled])
        acc_diffs.append(float(m_ok[idx].mean() - b_ok[idx].mean()))
        ll_diffs.append(float(ll_diff_i[idx].mean()))
    return {
        "accuracy_difference": float(m_ok.mean() - b_ok.mean()),
        "accuracy_difference_block_bootstrap_95ci": np.quantile(acc_diffs, [0.025, 0.975]).tolist(),
        "log_loss_difference": float(ll_diff_i.mean()),
        "log_loss_difference_block_bootstrap_95ci": np.quantile(ll_diffs, [0.025, 0.975]).tolist(),
        "model_only_correct": model_only,
        "market_only_correct": market_only,
        "mcnemar_exact_p": exact_p,
        "n_blocks": int(len(unique_blocks)),
    }


def main() -> None:
    os.makedirs(paths.RES, exist_ok=True)
    train = joblib.load(os.path.join(paths.PROCESSED, "train_dataset.pkl"))
    val = joblib.load(os.path.join(paths.PROCESSED, "val_dataset.pkl"))
    test = joblib.load(os.path.join(paths.PROCESSED, "test_dataset.pkl"))
    feat = pd.read_csv(
        os.path.join(paths.PROCESSED, "all_matches_featurized.csv"), parse_dates=["Date"]
    )
    meta = test["meta"].merge(
        feat[["Date", "HomeTeam", "AwayTeam", "mkt_prob_H", "mkt_prob_D", "mkt_prob_A"]],
        on=["Date", "HomeTeam", "AwayTeam"], how="left", validate="one_to_one",
    )
    y = np.asarray(test["target"], dtype=int)
    market = normalize_probabilities(
        meta[["mkt_prob_H", "mkt_prob_D", "mkt_prob_A"]].to_numpy(float)
    )
    iso = pd.to_datetime(meta["Date"]).dt.isocalendar()
    block = (
        meta["Div"].astype(str) + "_" + iso.year.astype(str) + "_" + iso.week.astype(str)
    ).to_numpy()

    predictions = {"xgboost": [], "random_forest": []}
    seed_rows = {"xgboost": [], "random_forest": []}
    for seed in SEEDS:
        xgb = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            early_stopping_rounds=30, random_state=seed, n_jobs=1,
        )
        xgb.fit(
            train["features"], train["target"],
            eval_set=[(val["features"], val["target"])], verbose=False,
        )
        px = normalize_probabilities(xgb.predict_proba(test["features"]))
        predictions["xgboost"].append(px)
        seed_rows["xgboost"].append({"seed": seed, **metrics(y, px)})

        rf = RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            random_state=seed, n_jobs=-1,
        )
        rf.fit(train["features"], train["target"])
        pr = normalize_probabilities(rf.predict_proba(test["features"]))
        predictions["random_forest"].append(pr)
        seed_rows["random_forest"].append({"seed": seed, **metrics(y, pr)})
        print(f"seed={seed} xgb={seed_rows['xgboost'][-1]} rf={seed_rows['random_forest'][-1]}")

    result = {"status": "complete", "n_test": int(len(y)), "seeds": list(SEEDS),
              "market": metrics(y, market), "models": {}}
    for name in predictions:
        ensemble = normalize_probabilities(np.mean(predictions[name], axis=0))
        result["models"][name] = {
            "per_seed": seed_rows[name],
            "accuracy_mean": float(np.mean([r["accuracy"] for r in seed_rows[name]])),
            "accuracy_sd": float(np.std([r["accuracy"] for r in seed_rows[name]], ddof=1)),
            "log_loss_mean": float(np.mean([r["log_loss"] for r in seed_rows[name]])),
            "log_loss_sd": float(np.std([r["log_loss"] for r in seed_rows[name]], ddof=1)),
            "ensemble": metrics(y, ensemble),
            "ensemble_vs_market": paired_summary(y, ensemble, market, block),
        }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
