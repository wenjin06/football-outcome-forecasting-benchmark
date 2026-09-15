"""
Leakage-controlled market-conditioned selective prediction prototype.

The outcome model is evaluated one season ahead.  A second-stage risk model
learns to predict whether the outcome model will be wrong, using only
information available before kickoff.  Its training labels come exclusively
from expanding-window out-of-fold predictions, never in-sample predictions.

This script is intentionally isolated from the published result pipeline.  It
writes results/selective_meta_prototype.json and does not overwrite any paper
table or existing result file.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import rel_entr
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import paths
from evaluate import financial_metrics
from risk import compute_scs, compute_ui, fit_robust


SEED = 42
OUT_PATH = os.path.join(paths.RES, "selective_meta_prototype.json")
DROP_COLS = ["Div", "Date", "Season", "HomeTeam", "AwayTeam", "FTR", "y"]
META_COLUMNS = [
    "p_model_h", "p_model_d", "p_model_a",
    "p_market_h", "p_market_d", "p_market_a",
    "model_pmax", "model_entropy", "model_margin",
    "market_pmax", "market_entropy", "market_margin",
    "tv_model_market", "kl_model_market", "argmax_disagree",
    "abs_diff_h", "abs_diff_d", "abs_diff_a",
    "close_vol", "move_h", "move_d", "move_a", "rank_gap",
]


@dataclass
class FoldPrediction:
    season: int
    frame: pd.DataFrame
    y: np.ndarray
    proba: np.ndarray


def make_outcome_model(seed: int = SEED) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        random_state=seed,
        n_jobs=-1,
    )


def fit_one_season_ahead(
    feat: pd.DataFrame,
    feature_cols: list[str],
    fit_through: int,
    validation_season: int,
    target_season: int,
    seed: int = SEED,
) -> FoldPrediction:
    train = feat[feat["Season"] <= fit_through]
    val = feat[feat["Season"] == validation_season]
    target = feat[feat["Season"] == target_season]
    if min(len(train), len(val), len(target)) == 0:
        raise ValueError(
            f"empty temporal fold train<={fit_through}, val={validation_season}, "
            f"target={target_season}"
        )
    model = make_outcome_model(seed)
    model.fit(
        train[feature_cols],
        train["y"].astype(int),
        eval_set=[(val[feature_cols], val["y"].astype(int))],
        verbose=False,
    )
    proba = model.predict_proba(target[feature_cols])
    return FoldPrediction(
        season=target_season,
        frame=target.copy(),
        y=target["y"].to_numpy(dtype=int),
        proba=proba,
    )


def normalized_entropy(p: np.ndarray) -> np.ndarray:
    return -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1) / np.log(p.shape[1])


def top_margin(p: np.ndarray) -> np.ndarray:
    ordered = np.sort(p, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def meta_features(fold: FoldPrediction) -> pd.DataFrame:
    f = fold.frame
    pm = f[["mkt_prob_H", "mkt_prob_D", "mkt_prob_A"]].to_numpy(dtype=float)
    pp = fold.proba
    tv = 0.5 * np.abs(pp - pm).sum(axis=1)
    kl = np.sum(rel_entr(np.clip(pp, 1e-12, 1), np.clip(pm, 1e-12, 1)), axis=1)
    out = pd.DataFrame({
        "p_model_h": pp[:, 0],
        "p_model_d": pp[:, 1],
        "p_model_a": pp[:, 2],
        "p_market_h": pm[:, 0],
        "p_market_d": pm[:, 1],
        "p_market_a": pm[:, 2],
        "model_pmax": pp.max(axis=1),
        "model_entropy": normalized_entropy(pp),
        "model_margin": top_margin(pp),
        "market_pmax": pm.max(axis=1),
        "market_entropy": normalized_entropy(pm),
        "market_margin": top_margin(pm),
        "tv_model_market": tv,
        "kl_model_market": kl,
        "argmax_disagree": (pp.argmax(axis=1) != pm.argmax(axis=1)).astype(float),
        "abs_diff_h": np.abs(pp[:, 0] - pm[:, 0]),
        "abs_diff_d": np.abs(pp[:, 1] - pm[:, 1]),
        "abs_diff_a": np.abs(pp[:, 2] - pm[:, 2]),
        "close_vol": f["close_vol"].to_numpy(dtype=float),
        "move_h": np.abs(f["odds_move_H"].to_numpy(dtype=float)),
        "move_d": np.abs(f["odds_move_D"].to_numpy(dtype=float)),
        "move_a": np.abs(f["odds_move_A"].to_numpy(dtype=float)),
        "rank_gap": np.abs(f["H_rank"].to_numpy(dtype=float) - f["A_rank"].to_numpy(dtype=float)),
    })
    return out.replace([np.inf, -np.inf], np.nan)


def augmented_features(
    fold: FoldPrediction, feature_cols: list[str]
) -> pd.DataFrame:
    """Meta-risk descriptors plus the original pre-match representation."""
    meta = meta_features(fold).reset_index(drop=True)
    original = fold.frame[feature_cols].reset_index(drop=True).copy()
    original.columns = [f"orig__{c}" for c in original.columns]
    return pd.concat([meta, original], axis=1)


def aurc(error: np.ndarray, risk: np.ndarray) -> float:
    order = np.argsort(risk, kind="stable")
    cumulative_risk = np.cumsum(error[order]) / np.arange(1, len(error) + 1)
    return float(cumulative_risk.mean())


def oracle_aurc(error: np.ndarray) -> float:
    return aurc(error, error.astype(float))


def risk_metrics(error: np.ndarray, risk: np.ndarray) -> dict:
    a = aurc(error, risk)
    return {
        "aurc": a,
        "eaurc": a - oracle_aurc(error),
        "auroc_error": float(roc_auc_score(error, risk)),
        "brier_error": float(brier_score_loss(error, np.clip(risk, 0, 1))),
        "logloss_error": float(log_loss(error, np.clip(risk, 1e-6, 1 - 1e-6))),
    }


def coverage_metrics(
    y: np.ndarray,
    proba: np.ndarray,
    risk: np.ndarray,
    odds: np.ndarray | None,
    coverages=(0.5, 0.7, 0.8, 0.9, 1.0),
) -> list[dict]:
    pred = proba.argmax(axis=1)
    rows = []
    for coverage in coverages:
        n_keep = max(1, int(round(len(y) * coverage)))
        keep = np.argsort(risk, kind="stable")[:n_keep]
        row = {
            "coverage": float(coverage),
            "n": int(n_keep),
            "accuracy": float((pred[keep] == y[keep]).mean()),
            "selective_risk": float((pred[keep] != y[keep]).mean()),
        }
        if odds is not None:
            chosen = odds[keep, pred[keep]]
            valid = np.isfinite(chosen) & (chosen > 1)
            returns = np.where(pred[keep][valid] == y[keep][valid], chosen[valid] - 1, -1.0)
            row.update(financial_metrics(returns))
        rows.append(row)
    return rows


def make_odds(frame: pd.DataFrame, raw: pd.DataFrame) -> np.ndarray:
    cols = ["Date", "HomeTeam", "AwayTeam", "B365CH", "B365CD", "B365CA"]
    merged = frame[["Date", "HomeTeam", "AwayTeam"]].merge(
        raw[cols], on=["Date", "HomeTeam", "AwayTeam"], how="left", validate="one_to_one"
    )
    return merged[["B365CH", "B365CD", "B365CA"]].to_numpy(dtype=float)


def bootstrap_aurc_difference(
    error: np.ndarray,
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    n = len(error)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = aurc(error[idx], risk_a[idx]) - aurc(error[idx], risk_b[idx])
    return {
        "difference_a_minus_b": float(aurc(error, risk_a) - aurc(error, risk_b)),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_a_not_better": float((diffs >= 0).mean()),
    }


def main() -> None:
    os.makedirs(paths.RES, exist_ok=True)
    feat = pd.read_csv(
        os.path.join(paths.PROCESSED, "all_matches_featurized.csv"),
        parse_dates=["Date"],
    )
    feat["Season"] = feat["Season"].astype(int)
    feature_cols = [
        c for c in feat.columns
        if c not in DROP_COLS and feat.loc[feat["Season"] <= 2023, c].notna().any()
    ]
    medians = feat.loc[feat["Season"] <= 2023, feature_cols].median()
    feat[feature_cols] = feat[feature_cols].fillna(medians)

    raw_files = [
        os.path.join(paths.raw_data_dir(), name)
        for name in os.listdir(paths.raw_data_dir()) if name.lower().endswith(".csv")
    ]
    raw = pd.concat([pd.read_csv(p) for p in raw_files], ignore_index=True)
    raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
    raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    raw = raw.drop_duplicates(["Date", "HomeTeam", "AwayTeam"], keep="last")

    fold_specs = [
        (2019, 2020, 2021),
        (2020, 2021, 2022),
        (2021, 2022, 2023),
        (2022, 2023, 2024),
        (2023, 2024, 2025),
    ]
    folds = []
    for fit_through, val_season, target_season in fold_specs:
        print(f"Outcome fold: train<= {fit_through}, val={val_season}, target={target_season}")
        folds.append(
            fit_one_season_ahead(
                feat, feature_cols, fit_through, val_season, target_season, SEED
            )
        )

    meta_train_folds = folds[:3]
    validation_fold = folds[3]
    test_fold = folds[4]
    x_train = pd.concat([meta_features(f) for f in meta_train_folds], ignore_index=True)
    x_train_full = pd.concat(
        [augmented_features(f, feature_cols) for f in meta_train_folds],
        ignore_index=True,
    )
    e_train = np.concatenate([
        (f.proba.argmax(axis=1) != f.y).astype(int) for f in meta_train_folds
    ])
    x_val = meta_features(validation_fold)
    x_val_full = augmented_features(validation_fold, feature_cols)
    e_val = (validation_fold.proba.argmax(axis=1) != validation_fold.y).astype(int)
    x_test = meta_features(test_fold)
    x_test_full = augmented_features(test_fold, feature_cols)
    e_test = (test_fold.proba.argmax(axis=1) != test_fold.y).astype(int)

    meta_medians = x_train.median()
    x_train = x_train.fillna(meta_medians)
    x_val = x_val.fillna(meta_medians)
    x_test = x_test.fillna(meta_medians)

    full_medians = x_train_full.median()
    x_train_full = x_train_full.fillna(full_medians)
    x_val_full = x_val_full.fillna(full_medians)
    x_test_full = x_test_full.fillna(full_medians)

    candidates = {
        "logistic_meta": (
            make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced", random_state=SEED),
            ),
            x_train[META_COLUMNS], x_val[META_COLUMNS], x_test[META_COLUMNS],
        ),
        "hist_meta": (
            HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                min_samples_leaf=40, l2_regularization=1.0, random_state=SEED,
            ),
            x_train[META_COLUMNS], x_val[META_COLUMNS], x_test[META_COLUMNS],
        ),
        "hist_augmented": (
            HistGradientBoostingClassifier(
                learning_rate=0.035, max_iter=250, max_leaf_nodes=15,
                min_samples_leaf=60, l2_regularization=2.0, random_state=SEED,
            ),
            x_train_full, x_val_full, x_test_full,
        ),
        "xgb_augmented": (
            XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.03,
                min_child_weight=20, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.5, reg_lambda=3.0, eval_metric="logloss",
                random_state=SEED, n_jobs=-1,
            ),
            x_train_full, x_val_full, x_test_full,
        ),
    }
    candidate_results = {}
    fitted = {}
    for name, (model, candidate_train, candidate_val, candidate_test) in candidates.items():
        model.fit(candidate_train, e_train)
        val_raw = model.predict_proba(candidate_val)[:, 1]
        candidate_results[name] = {
            "validation_aurc": aurc(e_val, val_raw),
            "validation_auroc_error": float(roc_auc_score(e_val, val_raw)),
        }
        fitted[name] = (model, val_raw, candidate_test)

    selected_name = min(candidate_results, key=lambda k: candidate_results[k]["validation_aurc"])
    selected_model, selected_val_raw, selected_test_features = fitted[selected_name]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(selected_val_raw, e_val)
    selected_test_raw = selected_model.predict_proba(selected_test_features)[:, 1]
    learned_risk = calibrator.predict(selected_test_raw)

    train_ref = feat[feat["Season"] <= 2023]
    vol_stats = fit_robust(train_ref, "close_vol")
    move_stats = fit_robust(train_ref.assign(
        move_total=train_ref["odds_move_H"].abs() + train_ref["odds_move_A"].abs()
    ), "move_total")
    baseline_scores = {
        "learned_risk": learned_risk,
        "1-pmax": 1.0 - test_fold.proba.max(axis=1),
        "entropy": normalized_entropy(test_fold.proba),
        "market_uncertainty": 1.0 - test_fold.frame["mkt_prob_max"].to_numpy(dtype=float),
        "UI": compute_ui(test_fold.proba, test_fold.frame, vol_stats, move_stats),
        "SCS": compute_scs(test_fold.frame, vol_stats, move_stats),
    }
    odds_test = make_odds(test_fold.frame, raw)
    methods = {}
    for name, risk in baseline_scores.items():
        metrics = risk_metrics(e_test, risk)
        methods[name] = {
            **metrics,
            "coverage": coverage_metrics(
                test_fold.y, test_fold.proba, risk, odds_test
            ),
        }

    comparisons = {
        "learned_vs_1-pmax": bootstrap_aurc_difference(
            e_test, learned_risk, baseline_scores["1-pmax"]
        ),
        "learned_vs_UI": bootstrap_aurc_difference(
            e_test, learned_risk, baseline_scores["UI"]
        ),
        "learned_vs_market_uncertainty": bootstrap_aurc_difference(
            e_test, learned_risk, baseline_scores["market_uncertainty"]
        ),
    }
    prob_true, prob_pred = calibration_curve(e_test, learned_risk, n_bins=10, strategy="quantile")
    result = {
        "status": "prototype",
        "method": "market-conditioned cross-fitted error-risk model",
        "seed": SEED,
        "temporal_folds": [
            {"fit_through": a, "validation": b, "target": c}
            for a, b, c in fold_specs
        ],
        "sample_sizes": {
            "meta_train": int(len(e_train)),
            "validation": int(len(e_val)),
            "test": int(len(e_test)),
        },
        "outcome_test": {
            "accuracy": float((test_fold.proba.argmax(axis=1) == test_fold.y).mean()),
            "log_loss": float(log_loss(test_fold.y, test_fold.proba)),
        },
        "risk_candidates_validation": candidate_results,
        "selected_risk_model": selected_name,
        "meta_features": META_COLUMNS,
        "learned_risk_calibration": {
            "mean_predicted_error": float(np.mean(learned_risk)),
            "observed_error": float(np.mean(e_test)),
            "bin_predicted": prob_pred.tolist(),
            "bin_observed": prob_true.tolist(),
        },
        "methods_test": methods,
        "paired_bootstrap_aurc": comparisons,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Selected risk model: {selected_name}")
    for name, row in methods.items():
        c70 = next(x for x in row["coverage"] if x["coverage"] == 0.7)
        print(
            f"{name:20s} AURC={row['aurc']:.4f} AUROC={row['auroc_error']:.4f} "
            f"Acc@70={c70['accuracy']:.4f} ROI@70={c70['roi']:.4f}"
        )
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
