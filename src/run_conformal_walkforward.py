"""Rolling-season evaluation of market-anchored conformal forecast triage.

For each target season t in 2022/23--2025/26:
  * outcome model training uses seasons <= t-3;
  * early stopping uses t-2;
  * the first half of t-1 selects a market/model convex blend;
  * the second half of t-1 calibrates conformal thresholds;
  * t is never touched until final evaluation.

The decision interface maps prediction-set size to three actions:
singleton = automated point forecast, doubleton = review, full set = abstain.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import paths
from run_conformal_selective import (
    conformal_quantile,
    lac_sets,
    normalize_probabilities,
)
from run_selective_meta import DROP_COLS, make_outcome_model


SEEDS = (7, 42, 2026)
TARGET_SEASONS = (2022, 2023, 2024, 2025)
ALPHAS = (0.10, 0.20, 0.30)
OUT_PATH = os.path.join(paths.RES, "conformal_walkforward.json")
PER_MATCH_PATH = os.path.join(paths.RES, "conformal_walkforward_final_per_match.csv")


def probabilities(frame: pd.DataFrame) -> np.ndarray:
    return normalize_probabilities(
        frame[["mkt_prob_H", "mkt_prob_D", "mkt_prob_A"]].to_numpy(float)
    )


def set_metrics(sets: np.ndarray, y: np.ndarray) -> dict:
    sizes = sets.sum(axis=1)
    contains = sets[np.arange(len(y)), y]
    actions = {
        "automated_singleton": sizes == 1,
        "review_doubleton": sizes == 2,
        "abstain_full_set": sizes == 3,
    }
    out = {
        "coverage": float(contains.mean()),
        "mean_set_size": float(sizes.mean()),
        "action_rates": {k: float(v.mean()) for k, v in actions.items()},
        "per_class_coverage": {},
    }
    for c, label in enumerate(("H", "D", "A")):
        mask = y == c
        out["per_class_coverage"][label] = float(contains[mask].mean())
    singleton = actions["automated_singleton"]
    out["singleton_n"] = int(singleton.sum())
    out["singleton_accuracy"] = (
        float((sets[singleton].argmax(axis=1) == y[singleton]).mean())
        if singleton.any() else None
    )
    return out


def calibrate_lac(p_cal: np.ndarray, y_cal: np.ndarray, alpha: float, mondrian: bool):
    if not mondrian:
        return conformal_quantile(1.0 - p_cal[np.arange(len(y_cal)), y_cal], alpha)
    return np.array([
        conformal_quantile(1.0 - p_cal[y_cal == c, c], alpha)
        for c in range(3)
    ])[None, :]


def adaptive_mondrian_sets(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    p_test: np.ndarray,
    y_test: np.ndarray,
    dates: np.ndarray,
    target_alpha: float,
    gamma: float,
) -> tuple[np.ndarray, dict]:
    """Prequential class-conditional LAC, updated only after each match day."""
    histories = [list(1.0 - p_cal[y_cal == c, c]) for c in range(3)]
    adaptive_alpha = np.full(3, target_alpha, dtype=float)
    sets = np.zeros_like(p_test, dtype=bool)
    alpha_trace = []
    for date in pd.unique(dates):
        idx = np.flatnonzero(dates == date)
        q = np.array([
            conformal_quantile(np.asarray(histories[c]), adaptive_alpha[c])
            for c in range(3)
        ])[None, :]
        sets[idx] = lac_sets(p_test[idx], q)
        # Outcomes from a date update the state only after all matches on that
        # date were predicted, avoiding within-day outcome leakage.
        for i in idx:
            c = int(y_test[i])
            error = float(not sets[i, c])
            adaptive_alpha[c] = np.clip(
                adaptive_alpha[c] + gamma * (target_alpha - error), 0.01, 0.50
            )
            histories[c].append(float(1.0 - p_test[i, c]))
        alpha_trace.append(adaptive_alpha.copy())
    trace = np.asarray(alpha_trace)
    return sets, {
        "gamma": gamma,
        "final_alpha_by_class": adaptive_alpha.tolist(),
        "mean_alpha_by_class": trace.mean(axis=0).tolist(),
    }


def main() -> None:
    os.makedirs(paths.RES, exist_ok=True)
    feat = pd.read_csv(
        os.path.join(paths.PROCESSED, "all_matches_featurized.csv"),
        parse_dates=["Date"],
    )
    feat["Season"] = feat["Season"].astype(int)
    rows = []
    per_match_rows = []

    for target in TARGET_SEASONS:
        train = feat[feat["Season"] <= target - 3].copy()
        early = feat[feat["Season"] == target - 2].copy()
        previous = feat[feat["Season"] == target - 1].sort_values("Date").copy()
        test = feat[feat["Season"] == target].sort_values("Date").copy()
        split = len(previous) // 2
        tune, calibration = previous.iloc[:split], previous.iloc[split:]
        feature_cols = [
            c for c in feat.columns
            if c not in DROP_COLS and train[c].notna().any()
        ]
        medians = train[feature_cols].median()
        for frame in (train, early, tune, calibration, test):
            frame.loc[:, feature_cols] = frame[feature_cols].fillna(medians)

        market_tune = probabilities(tune)
        market_cal = probabilities(calibration)
        market_test = probabilities(test)
        y_tune = tune["y"].to_numpy(int)
        y_cal = calibration["y"].to_numpy(int)
        y_test = test["y"].to_numpy(int)

        for seed in SEEDS:
            model = make_outcome_model(seed)
            model.fit(
                train[feature_cols], train["y"].astype(int),
                eval_set=[(early[feature_cols], early["y"].astype(int))],
                verbose=False,
            )
            model_tune = normalize_probabilities(model.predict_proba(tune[feature_cols]))
            model_cal = normalize_probabilities(model.predict_proba(calibration[feature_cols]))
            model_test = normalize_probabilities(model.predict_proba(test[feature_cols]))

            grid = np.linspace(0.0, 1.0, 21)
            losses = [
                log_loss(y_tune, w * model_tune + (1.0 - w) * market_tune)
                for w in grid
            ]
            weight = float(grid[int(np.argmin(losses))])
            blend_cal = weight * model_cal + (1.0 - weight) * market_cal
            blend_test = weight * model_test + (1.0 - weight) * market_test

            for method, p_cal, p_test in (
                ("market", market_cal, market_test),
                ("model", model_cal, model_test),
                ("selected_blend", blend_cal, blend_test),
            ):
                for alpha in ALPHAS:
                    for mondrian in (False, True):
                        qhat = calibrate_lac(p_cal, y_cal, alpha, mondrian)
                        fixed_sets = lac_sets(p_test, qhat)
                        metrics = set_metrics(fixed_sets, y_test)
                        rows.append({
                            "target_season": target,
                            "seed": seed,
                            "method": method,
                            "calibration": "mondrian" if mondrian else "marginal",
                            "alpha": alpha,
                            "selected_model_weight": weight,
                            "point_accuracy": float((p_test.argmax(axis=1) == y_test).mean()),
                            "point_log_loss": float(log_loss(y_test, p_test)),
                            "n_test": int(len(test)),
                            **metrics,
                        })
                        if target == 2025 and seed == 42 and method == "selected_blend":
                            for i, (_, match) in enumerate(test.iterrows()):
                                per_match_rows.append({
                                    "Date": match["Date"], "Div": match["Div"],
                                    "HomeTeam": match["HomeTeam"], "AwayTeam": match["AwayTeam"],
                                    "y": int(y_test[i]), "alpha": alpha,
                                    "calibration": "mondrian" if mondrian else "marginal",
                                    "set_H": int(fixed_sets[i, 0]), "set_D": int(fixed_sets[i, 1]),
                                    "set_A": int(fixed_sets[i, 2]),
                                })
                    for gamma in (0.0, 0.005, 0.01, 0.02, 0.05):
                        sets, adaptive_state = adaptive_mondrian_sets(
                            p_cal, y_cal, p_test, y_test,
                            test["Date"].dt.strftime("%Y-%m-%d").to_numpy(),
                            alpha, gamma,
                        )
                        rows.append({
                            "target_season": target,
                            "seed": seed,
                            "method": method,
                            "calibration": f"adaptive_mondrian_g{gamma}",
                            "alpha": alpha,
                            "selected_model_weight": weight,
                            "point_accuracy": float((p_test.argmax(axis=1) == y_test).mean()),
                            "point_log_loss": float(log_loss(y_test, p_test)),
                            "n_test": int(len(test)),
                            "adaptive_state": adaptive_state,
                            **set_metrics(sets, y_test),
                        })
                        if (target == 2025 and seed == 42 and method == "selected_blend"
                                and gamma == 0.02):
                            for i, (_, match) in enumerate(test.iterrows()):
                                per_match_rows.append({
                                    "Date": match["Date"], "Div": match["Div"],
                                    "HomeTeam": match["HomeTeam"], "AwayTeam": match["AwayTeam"],
                                    "y": int(y_test[i]), "alpha": alpha,
                                    "calibration": "adaptive_mondrian_g0.02",
                                    "set_H": int(sets[i, 0]), "set_D": int(sets[i, 1]),
                                    "set_A": int(sets[i, 2]),
                                })

    result = {
        "status": "complete",
        "protocol": "rolling four-way temporal split",
        "seeds": list(SEEDS),
        "rows": rows,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    pd.DataFrame(per_match_rows).to_csv(PER_MATCH_PATH, index=False)

    summary = pd.DataFrame(rows)
    view = summary[
        (summary["alpha"] == 0.10) & (summary["calibration"] == "mondrian")
    ]
    print("alpha=.10 Mondrian-LAC, mean across season-seed runs")
    for method, group in view.groupby("method"):
        print(
            f"{method:16s} cov={group.coverage.mean():.3f} "
            f"size={group.mean_set_size.mean():.3f} "
            f"singleton={group.action_rates.map(lambda x: x['automated_singleton']).mean():.3f} "
            f"sacc={group.singleton_accuracy.mean():.3f} "
            f"draw_cov={group.per_class_coverage.map(lambda x: x['D']).mean():.3f} "
            f"weight={group.selected_model_weight.mean():.3f}"
        )
    print(f"Saved: {OUT_PATH}")
    print(f"Saved: {PER_MATCH_PATH}")


if __name__ == "__main__":
    main()
