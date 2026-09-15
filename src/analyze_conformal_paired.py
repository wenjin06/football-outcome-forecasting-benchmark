"""Dependence-aware paired bootstrap for final-season conformal set policies."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import paths


IN_PATH = os.path.join(paths.RES, "conformal_walkforward_final_per_match.csv")
OUT_PATH = os.path.join(paths.RES, "conformal_paired_bootstrap.json")


def metrics(y: np.ndarray, sets: np.ndarray, target: float) -> dict:
    contains = sets[np.arange(len(y)), y]
    sizes = sets.sum(axis=1)
    coverages = [contains[y == c].mean() for c in range(3)]
    singleton = sizes == 1
    return {
        "coverage": float(contains.mean()),
        "absolute_target_error": float(abs(contains.mean() - target)),
        "class_coverage_gap": float(max(coverages) - min(coverages)),
        "mean_set_size": float(sizes.mean()),
        "singleton_rate": float(singleton.mean()),
        "singleton_accuracy": float((sets[singleton].argmax(axis=1) == y[singleton]).mean()),
    }


def main() -> None:
    df = pd.read_csv(IN_PATH, parse_dates=["Date"])
    rng = np.random.default_rng(42)
    results = {}
    keys = ["Date", "Div", "HomeTeam", "AwayTeam", "y"]
    for alpha in (0.10, 0.20, 0.30):
        frames = {}
        for calibration in ("marginal", "mondrian", "adaptive_mondrian_g0.02"):
            frame = df[(df["alpha"] == alpha) & (df["calibration"] == calibration)].copy()
            frame = frame.sort_values(keys).reset_index(drop=True)
            frames[calibration] = frame
        reference = frames["adaptive_mondrian_g0.02"]
        for name, frame in frames.items():
            if not reference[keys].equals(frame[keys]):
                raise RuntimeError(f"Per-match alignment failed for {alpha} {name}")
        y = reference["y"].to_numpy(int)
        block = (
            reference["Div"].astype(str) + "_" +
            reference["Date"].dt.isocalendar().year.astype(str) + "_" +
            reference["Date"].dt.isocalendar().week.astype(str)
        ).to_numpy()
        unique_blocks = np.unique(block)
        members = {b: np.flatnonzero(block == b) for b in unique_blocks}
        sets_by_method = {
            name: frame[["set_H", "set_D", "set_A"]].to_numpy(bool)
            for name, frame in frames.items()
        }
        target = 1.0 - alpha
        point = {name: metrics(y, sets, target) for name, sets in sets_by_method.items()}
        comparisons = {}
        for baseline in ("marginal", "mondrian"):
            draws = {k: [] for k in (
                "target_error_reduction", "class_gap_reduction", "set_size_reduction",
                "singleton_rate_change", "singleton_accuracy_change",
            )}
            for _ in range(5000):
                sampled = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
                idx = np.concatenate([members[b] for b in sampled])
                adaptive = metrics(y[idx], sets_by_method["adaptive_mondrian_g0.02"][idx], target)
                base = metrics(y[idx], sets_by_method[baseline][idx], target)
                draws["target_error_reduction"].append(
                    base["absolute_target_error"] - adaptive["absolute_target_error"]
                )
                draws["class_gap_reduction"].append(
                    base["class_coverage_gap"] - adaptive["class_coverage_gap"]
                )
                draws["set_size_reduction"].append(
                    base["mean_set_size"] - adaptive["mean_set_size"]
                )
                draws["singleton_rate_change"].append(
                    adaptive["singleton_rate"] - base["singleton_rate"]
                )
                draws["singleton_accuracy_change"].append(
                    adaptive["singleton_accuracy"] - base["singleton_accuracy"]
                )
            comparisons[f"adaptive_vs_{baseline}"] = {
                key: {
                    "difference": float(np.mean(values)),
                    "block_bootstrap_95ci": np.quantile(values, [0.025, 0.975]).tolist(),
                    "probability_improvement_le_zero": float((np.asarray(values) <= 0).mean()),
                }
                for key, values in draws.items()
            }
        results[f"alpha_{alpha:.2f}"] = {
            "target_coverage": target,
            "n": int(len(y)),
            "n_blocks": int(len(unique_blocks)),
            "methods": point,
            "comparisons": comparisons,
        }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results["alpha_0.10"], indent=2))
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
