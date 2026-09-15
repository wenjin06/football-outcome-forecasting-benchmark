"""Market-anchored conformal prediction sets for selective football forecasts.

The script uses a strict four-way temporal protocol:
  outcome training: 2019/20--2022/23
  early stopping:    2023/24
  blend selection:   first half of 2024/25
  conformal calib.:  second half of 2024/25
  final test:        complete 2025/26

It compares marginal LAC and class-conditional (Mondrian) LAC sets for the
model probabilities, market probabilities, and a validation-selected convex
blend.  Existing paper results are never overwritten.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import paths
from evaluate import financial_metrics
from run_selective_meta import DROP_COLS, make_odds, make_outcome_model


SEED = 42
OUT_PATH = os.path.join(paths.RES, "conformal_selective_prototype.json")
PER_MATCH_PATH = os.path.join(paths.RES, "conformal_selective_per_match.csv")


def json_safe(value):
    """Recursively replace non-finite floats with ``None``.

    ``json.dump`` writes bare ``NaN``/``Infinity`` tokens by default, which are
    not valid JSON and break any downstream reader. Degenerate subgroups (for
    example a singleton subset holding a single match) can produce a Sharpe
    ratio with zero variance, so those values must be neutralised explicitly.
    """
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    k = min(max(k, 1), len(scores))
    return float(np.partition(scores, k - 1)[k - 1])


def lac_sets(p: np.ndarray, q: float | np.ndarray) -> np.ndarray:
    return (1.0 - p) <= q


def normalize_probabilities(p: np.ndarray) -> np.ndarray:
    """Clip numerical noise and enforce the simplex before scoring."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def aps_scores(p: np.ndarray, y: np.ndarray, lam: float = 0.0, k_reg: int = 1) -> np.ndarray:
    """Deterministic APS/RAPS nonconformity score for the observed class."""
    order = np.argsort(-p, axis=1, kind="stable")
    ranked = np.take_along_axis(p, order, axis=1)
    ranks = np.argsort(order, axis=1, kind="stable")
    cumulative = np.cumsum(ranked, axis=1)
    row = np.arange(len(p))
    true_rank = ranks[row, y]
    penalty = lam * np.maximum(true_rank + 1 - k_reg, 0)
    return cumulative[row, true_rank] + penalty


def aps_sets(p: np.ndarray, q: float | np.ndarray, lam: float = 0.0, k_reg: int = 1) -> np.ndarray:
    """Invert deterministic APS/RAPS scores for every candidate class."""
    order = np.argsort(-p, axis=1, kind="stable")
    ranked = np.take_along_axis(p, order, axis=1)
    cumulative = np.cumsum(ranked, axis=1)
    rank_numbers = np.arange(1, p.shape[1] + 1)[None, :]
    ranked_scores = cumulative + lam * np.maximum(rank_numbers - k_reg, 0)
    q_arr = np.asarray(q)
    if q_arr.ndim == 0:
        ranked_q = q_arr
    else:
        class_q = np.broadcast_to(q_arr, p.shape)
        ranked_q = np.take_along_axis(class_q, order, axis=1)
    ranked_sets = ranked_scores <= ranked_q
    # Always retain the top-ranked class; otherwise very small quantiles can
    # generate an empty prediction set, which is not actionable.
    ranked_sets[:, 0] = True
    sets = np.zeros_like(ranked_sets, dtype=bool)
    np.put_along_axis(sets, order, ranked_sets, axis=1)
    return sets


def evaluate_sets(
    sets: np.ndarray,
    y: np.ndarray,
    point_pred: np.ndarray,
    odds: np.ndarray,
) -> dict:
    contains = sets[np.arange(len(y)), y]
    sizes = sets.sum(axis=1)
    singleton = sizes == 1
    per_class = {}
    for c, label in enumerate(("H", "D", "A")):
        mask = y == c
        per_class[label] = {
            "n": int(mask.sum()),
            "coverage": float(contains[mask].mean()),
            "mean_set_size": float(sizes[mask].mean()),
        }
    result = {
        "marginal_coverage": float(contains.mean()),
        "mean_set_size": float(sizes.mean()),
        "singleton_rate": float(singleton.mean()),
        "singleton_n": int(singleton.sum()),
        "per_class": per_class,
    }
    if singleton.any():
        selected = np.flatnonzero(singleton)
        pred = point_pred[selected]
        result["singleton_accuracy"] = float((pred == y[selected]).mean())
        chosen_odds = odds[selected, pred]
        valid = np.isfinite(chosen_odds) & (chosen_odds > 1)
        returns = np.where(
            pred[valid] == y[selected][valid], chosen_odds[valid] - 1.0, -1.0
        )
        result["singleton_financial"] = financial_metrics(returns)
    return result


def method_results(
    p_tune: np.ndarray,
    y_tune: np.ndarray,
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    p_test: np.ndarray,
    y_test: np.ndarray,
    odds_test: np.ndarray,
) -> dict:
    del p_tune, y_tune
    point_pred = p_test.argmax(axis=1)
    out = {}
    for alpha in (0.05, 0.10, 0.20, 0.30):
        marginal_q = conformal_quantile(1.0 - p_cal[np.arange(len(y_cal)), y_cal], alpha)
        marginal_sets = lac_sets(p_test, marginal_q)

        class_q = np.empty(3)
        for c in range(3):
            mask = y_cal == c
            class_q[c] = conformal_quantile(1.0 - p_cal[mask, c], alpha)
        mondrian_sets = lac_sets(p_test, class_q[None, :])
        method_row = {
            "target_coverage": 1.0 - alpha,
            "marginal_lac": {
                "qhat": marginal_q,
                **evaluate_sets(marginal_sets, y_test, point_pred, odds_test),
            },
            "mondrian_lac": {
                "qhat_by_class": class_q.tolist(),
                **evaluate_sets(mondrian_sets, y_test, point_pred, odds_test),
            },
        }
        for score_name, lam in (("aps", 0.0), ("raps", 0.05)):
            cal_scores = aps_scores(p_cal, y_cal, lam=lam)
            qhat = conformal_quantile(cal_scores, alpha)
            sets = aps_sets(p_test, qhat, lam=lam)
            class_q_aps = np.empty(3)
            for c in range(3):
                mask = y_cal == c
                class_q_aps[c] = conformal_quantile(cal_scores[mask], alpha)
            mondrian_sets_aps = aps_sets(p_test, class_q_aps[None, :], lam=lam)
            method_row[score_name] = {
                "qhat": qhat,
                **evaluate_sets(sets, y_test, point_pred, odds_test),
            }
            method_row[f"mondrian_{score_name}"] = {
                "qhat_by_class": class_q_aps.tolist(),
                **evaluate_sets(mondrian_sets_aps, y_test, point_pred, odds_test),
            }
        out[f"alpha_{alpha:.2f}"] = method_row
    return out


def main() -> None:
    os.makedirs(paths.RES, exist_ok=True)
    feat = pd.read_csv(
        os.path.join(paths.PROCESSED, "all_matches_featurized.csv"),
        parse_dates=["Date"],
    )
    feat["Season"] = feat["Season"].astype(int)
    feature_cols = [
        c for c in feat.columns
        if c not in DROP_COLS and feat.loc[feat["Season"] <= 2022, c].notna().any()
    ]
    medians = feat.loc[feat["Season"] <= 2022, feature_cols].median()
    feat[feature_cols] = feat[feature_cols].fillna(medians)

    train = feat[feat["Season"] <= 2022]
    early_stop = feat[feat["Season"] == 2023]
    validation = feat[feat["Season"] == 2024].sort_values("Date")
    test = feat[feat["Season"] == 2025].sort_values("Date")
    split = len(validation) // 2
    tune = validation.iloc[:split]
    calibration = validation.iloc[split:]

    model = make_outcome_model(SEED)
    model.fit(
        train[feature_cols], train["y"].astype(int),
        eval_set=[(early_stop[feature_cols], early_stop["y"].astype(int))],
        verbose=False,
    )
    p_tune_model = normalize_probabilities(model.predict_proba(tune[feature_cols]))
    p_cal_model = normalize_probabilities(model.predict_proba(calibration[feature_cols]))
    p_test_model = normalize_probabilities(model.predict_proba(test[feature_cols]))
    p_tune_market = normalize_probabilities(
        tune[["mkt_prob_H", "mkt_prob_D", "mkt_prob_A"]].to_numpy(float)
    )
    p_cal_market = normalize_probabilities(
        calibration[["mkt_prob_H", "mkt_prob_D", "mkt_prob_A"]].to_numpy(float)
    )
    p_test_market = normalize_probabilities(
        test[["mkt_prob_H", "mkt_prob_D", "mkt_prob_A"]].to_numpy(float)
    )
    y_tune = tune["y"].to_numpy(int)
    y_cal = calibration["y"].to_numpy(int)
    y_test = test["y"].to_numpy(int)

    grid = np.linspace(0.0, 1.0, 21)
    grid_rows = []
    for model_weight in grid:
        p = model_weight * p_tune_model + (1 - model_weight) * p_tune_market
        grid_rows.append({
            "model_weight": float(model_weight),
            "log_loss": float(log_loss(y_tune, p)),
        })
    best = min(grid_rows, key=lambda row: row["log_loss"])
    w = best["model_weight"]
    p_tune_blend = w * p_tune_model + (1 - w) * p_tune_market
    p_cal_blend = w * p_cal_model + (1 - w) * p_cal_market
    p_test_blend = w * p_test_model + (1 - w) * p_test_market

    raw_files = [
        os.path.join(paths.raw_data_dir(), name)
        for name in os.listdir(paths.raw_data_dir()) if name.lower().endswith(".csv")
    ]
    raw = pd.concat([pd.read_csv(p) for p in raw_files], ignore_index=True)
    raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
    raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    raw = raw.drop_duplicates(["Date", "HomeTeam", "AwayTeam"], keep="last")
    odds_test = make_odds(test, raw)

    methods = {}
    for name, pt, pc, pte in (
        ("model", p_tune_model, p_cal_model, p_test_model),
        ("market", p_tune_market, p_cal_market, p_test_market),
        ("validation_selected_blend", p_tune_blend, p_cal_blend, p_test_blend),
    ):
        methods[name] = {
            "point_accuracy": float((pte.argmax(axis=1) == y_test).mean()),
            "point_log_loss": float(log_loss(y_test, pte)),
            "sets": method_results(pt, y_tune, pc, y_cal, pte, y_test, odds_test),
        }

    # A transparent decision baseline: issue a point decision only when both
    # the market and learned model produce the same singleton Mondrian-LAC set.
    alpha = 0.10
    model_q = np.array([
        conformal_quantile(1.0 - p_cal_model[y_cal == c, c], alpha) for c in range(3)
    ])
    market_q = np.array([
        conformal_quantile(1.0 - p_cal_market[y_cal == c, c], alpha) for c in range(3)
    ])
    model_sets = lac_sets(p_test_model, model_q[None, :])
    market_sets = lac_sets(p_test_market, market_q[None, :])
    consensus = (model_sets.sum(axis=1) == 1) & (market_sets.sum(axis=1) == 1)
    consensus &= model_sets.argmax(axis=1) == market_sets.argmax(axis=1)
    consensus_sets = np.zeros_like(model_sets)
    consensus_sets[consensus] = model_sets[consensus]
    methods["market_model_consensus"] = {
        "alpha": alpha,
        "selection_rate": float(consensus.mean()),
        **evaluate_sets(
            consensus_sets, y_test, p_test_market.argmax(axis=1), odds_test
        ),
    }

    export = test[["Date", "Season", "Div", "HomeTeam", "AwayTeam", "y"]].copy()
    for label, probs in (("model", p_test_model), ("market", p_test_market)):
        for c, outcome in enumerate(("H", "D", "A")):
            export[f"{label}_p_{outcome}"] = probs[:, c]
        for alpha in (0.10, 0.20, 0.30):
            q = np.array([
                conformal_quantile(
                    1.0 - (p_cal_model if label == "model" else p_cal_market)[y_cal == c, c],
                    alpha,
                )
                for c in range(3)
            ])
            sets = lac_sets(probs, q[None, :])
            export[f"{label}_mondrian_size_a{int(alpha * 100):02d}"] = sets.sum(axis=1)
            export[f"{label}_mondrian_mask_a{int(alpha * 100):02d}"] = [
                "".join(np.array(["H", "D", "A"])[row]) for row in sets
            ]
    export.to_csv(PER_MATCH_PATH, index=False)

    result = {
        "status": "prototype",
        "protocol": {
            "outcome_train_through": 2022,
            "early_stopping_season": 2023,
            "blend_tuning": "first chronological half of 2024/25",
            "conformal_calibration": "second chronological half of 2024/25",
            "test": "complete 2025/26",
        },
        "sample_sizes": {
            "outcome_train": int(len(train)),
            "early_stop": int(len(early_stop)),
            "blend_tune": int(len(tune)),
            "conformal_calibration": int(len(calibration)),
            "test": int(len(test)),
        },
        "blend_grid": grid_rows,
        "selected_model_weight": w,
        "methods": methods,
    }
    result = json_safe(result)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Selected model weight: {w:.2f}")
    for method_name, method in methods.items():
        if "sets" not in method:
            print(
                f"{method_name:28s} selection={method['selection_rate']:.3f} "
                f"accuracy={method['singleton_accuracy']:.3f}"
            )
            continue
        row = method["sets"]["alpha_0.10"]
        marginal = row["marginal_lac"]
        mondrian = row["mondrian_lac"]
        print(
            f"{method_name:28s} point_acc={method['point_accuracy']:.3f} "
            f"marg_cov={marginal['marginal_coverage']:.3f} "
            f"marg_size={marginal['mean_set_size']:.3f} "
            f"mond_cov={mondrian['marginal_coverage']:.3f} "
            f"draw_cov={mondrian['per_class']['D']['coverage']:.3f} "
            f"singleton={mondrian['singleton_rate']:.3f}"
        )
    print(f"Saved: {OUT_PATH}")
    print(f"Saved: {PER_MATCH_PATH}")


if __name__ == "__main__":
    main()
