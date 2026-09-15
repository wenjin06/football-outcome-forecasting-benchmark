"""Generate audited LaTeX tables and figures for the upgraded manuscript."""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths


PAPER = os.path.join(paths.BASE, "paper-upgrade")
TABLES = os.path.join(PAPER, "tables")
FIGURES = os.path.join(PAPER, "figures")


def load(name):
    with open(os.path.join(paths.RES, name), encoding="utf-8") as f:
        return json.load(f)


def pct(x, digits=1):
    return f"{100*x:.{digits}f}\\%"


def save_table(name, caption, label, columns, rows, align=None, font=None,
               tabcolsep="5pt"):
    if align is None:
        align = "l" + "c" * (len(columns) - 1)
    if font is None:
        # Wide tables get a smaller base size instead of being scaled to the
        # text width: \resizebox distorts glyph sizes and makes adjacent tables
        # inconsistent.
        font = r"\footnotesize" if len(columns) >= 6 else r"\small"
    body = [
        "\\begin{table}[tbp]", "\\centering", font,
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule", " & ".join(columns) + " \\\\", "\\midrule",
    ]
    body += [" & ".join(map(str, row)) + " \\\\" for row in rows]
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(os.path.join(TABLES, name), "w", encoding="utf-8") as f:
        f.write("\n".join(body))


def main():
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    baselines = load("baselines_summary.json")
    multi = load("multiseed_paired.json")
    ablation = load("ablations_summary.json")
    walk = load("conformal_walkforward.json")
    paired = load("conformal_paired_bootstrap.json")
    llm = load("llm_conformal_triage.json")
    value = load("value_betting.json")

    # Table 1: complete-season predictive benchmark.
    rows = [["Market (de-vigged closing odds)", pct(multi["market"]["accuracy"]),
             f'{multi["market"]["log_loss"]:.3f}', "reference", "---"]]
    for key, label in (("xgboost", "XGBoost, five-seed ensemble"),
                       ("random_forest", "Random forest, five-seed ensemble")):
        model = multi["models"][key]
        comp = model["ensemble_vs_market"]
        ci = comp["accuracy_difference_block_bootstrap_95ci"]
        rows.append([label, pct(model["ensemble"]["accuracy"]),
                     f'{model["ensemble"]["log_loss"]:.3f}',
                     f'[{100*ci[0]:.2f}, {100*ci[1]:.2f}] pp',
                     f'{comp["mcnemar_exact_p"]:.3f}'])
    for key, label in (("elo", "Elo"), ("poisson", "Poisson")):
        rows.append([label, pct(baselines[key]["accuracy"]),
                     f'{baselines[key]["log_loss"]:.3f}', "---", "---"])
    save_table(
        "table1_predictive.tex",
        "Predictive performance on the complete 2025/26 test season (1,752 matches). Confidence intervals are league-by-week block-bootstrap intervals for the accuracy difference relative to the market; McNemar tests use paired classifications.",
        "tab:predictive", ["Method", "Accuracy", "Log loss", "$\\Delta$ accuracy 95\\% CI", "McNemar $p$"], rows,
    )

    # Table 2: complete-season ablation.
    names = [("full", "Full (60 features)"), ("no_odds", "Without market odds"),
             ("no_xg", "Without xG"), ("no_referee", "Without referee"),
             ("no_rank", "Without rank"), ("no_form", "Without form"),
             ("structured_only", "Structured non-market only")]
    rows = [[label, v["n_features"], pct(v["accuracy"]), f'{v["log_loss"]:.3f}',
             pct(v["roi_all"])] for key, label in names for v in [ablation[key]]]
    save_table(
        "table2_ablation.tex",
        "Feature-group ablation on the complete 2025/26 season. ROI is descriptive and uses the common closing-line settlement protocol.",
        "tab:ablation", ["Configuration", "Features", "Accuracy", "Log loss", "ROI"], rows,
    )

    wf = pd.DataFrame(walk["rows"])
    wf = wf[(wf.method == "selected_blend") & (wf.alpha == 0.10)]
    rows = []
    for cal, label in (("marginal", "Marginal LAC"), ("mondrian", "Fixed Mondrian-LAC"),
                       ("adaptive_mondrian_g0.0", "Expanding Mondrian ($\\gamma=0$)"),
                       ("adaptive_mondrian_g0.02", "Adaptive Mondrian ($\\gamma=0.02$)")):
        g = wf[wf.calibration == cal]
        gaps = g.per_class_coverage.map(lambda x: max(x.values()) - min(x.values()))
        errors = (g.coverage - 0.90).abs()
        single = g.action_rates.map(lambda x: x["automated_singleton"])
        rows.append([label, pct(g.coverage.mean()), pct(errors.mean(), 2), pct(gaps.mean(), 2),
                     f'{g.mean_set_size.mean():.3f}', pct(single.mean()), pct(g.singleton_accuracy.mean())])
    save_table(
        "table3_conformal_rolling.tex",
        "Rolling evaluation over the 2022/23--2025/26 target seasons at 90\\% target coverage (mean across season--seed runs). Target error is $|\\widehat{C}-0.90|$; class gap is the maximum minus minimum H/D/A coverage.",
        "tab:conformal-rolling",
        ["Set method", "Coverage", "Target error", "Class gap", "Mean size", "Automated", "Auto. acc."], rows,
    )

    # Table 4: final-season operating points.
    rows = []
    for alpha in (0.10, 0.20, 0.30):
        g = pd.DataFrame(walk["rows"])
        g = g[(g.method == "selected_blend") & (g.target_season == 2025) &
              (g.alpha == alpha) & (g.calibration == "adaptive_mondrian_g0.02")]
        x = g.iloc[0]
        actions = x.action_rates
        cls = x.per_class_coverage
        rows.append([pct(1-alpha, 0), pct(x.coverage),
                     f'{pct(cls["H"])} / {pct(cls["D"])} / {pct(cls["A"])}',
                     pct(actions["automated_singleton"]), pct(actions["review_doubleton"]),
                     pct(actions["abstain_full_set"]), pct(x.singleton_accuracy)])
    save_table(
        "table4_operating_points.tex",
        "Operating points of adaptive Mondrian-LAC on the complete 2025/26 season. Set size one is automated, size two is routed to review, and size three is an abstention.",
        "tab:operating", ["Target", "Coverage", "H / D / A coverage", "Automate", "Review", "Abstain", "Auto. acc."], rows,
    )

    # Table 5: paired bootstrap comparison at the primary target.
    p10 = paired["alpha_0.10"]
    rows = []
    metric_labels = [
        ("target_error_reduction", "Target-error reduction"),
        ("class_gap_reduction", "Class-gap reduction"),
        ("set_size_reduction", "Set-size reduction"),
        ("singleton_rate_change", "Automation-rate change"),
        ("singleton_accuracy_change", "Automated-accuracy change"),
    ]
    for comp_key, comp_label in (("adaptive_vs_marginal", "vs marginal LAC"),
                                 ("adaptive_vs_mondrian", "vs fixed Mondrian")):
        comp = p10["comparisons"][comp_key]
        for key, label in metric_labels:
            x = comp[key]
            ci = x["block_bootstrap_95ci"]
            scale = 1.0 if key == "set_size_reduction" else 100.0
            suffix = " classes" if key == "set_size_reduction" else " pp"
            rows.append([comp_label, label, f'{scale*x["difference"]:.2f}{suffix}',
                         f'[{scale*ci[0]:.2f}, {scale*ci[1]:.2f}]'])
    save_table(
        "table5_paired_conformal.tex",
        "Dependence-aware paired comparisons on the complete 2025/26 season at 90\\% target coverage (5,000 league-by-week block-bootstrap replicates). Positive values favor the adaptive method except where the metric explicitly reports an accuracy change.",
        "tab:paired-conformal", ["Comparison", "Metric", "Difference", "95\\% CI"], rows,
    )

    # Table 6: LLM audit on the frozen subset.
    rows = []
    for x in llm["rows"]:
        if abs(x["alpha"] - 0.10) < 1e-9:
            rows.append([x["action"].capitalize(), x["n"], pct(x["share"]),
                         pct(x["market_accuracy"]), pct(x["llm_accuracy"]),
                         pct(x["llm_restricted_accuracy"]),
                         f'{x["market_vs_llm_restricted"]["mcnemar_exact_p"]:.3f}'])
    save_table(
        "table6_llm_triage.tex",
        "Incremental LLM audit on the existing 1,104-match subset at the 90\\% conformal operating point. Restricted LLM chooses its highest-probability label within the market prediction set; no new API calls were made.",
        "tab:llm-triage", ["Triage action", "$N$", "Share", "Market acc.", "LLM acc.", "Restricted LLM", "$p$"], rows,
    )

    # Table 7: value selection falsification.
    rows = [[f'{x["threshold"]:.2f}', pct(x["roi"]), x["n_bets"], pct(x["coverage"])]
            for x in value["value_ev_scan"]]
    save_table(
        "table7_value.tex",
        "Value-based selection on the complete 2025/26 season. Every threshold produces a negative return against closing odds.",
        "tab:value", ["EV threshold", "ROI", "Bets", "Coverage"], rows,
    )

    # Figures.
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    final = pd.read_csv(os.path.join(paths.RES, "conformal_walkforward_final_per_match.csv"), parse_dates=["Date"])
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    methods = [("marginal", "Marginal LAC"), ("mondrian", "Fixed Mondrian"),
               ("adaptive_mondrian_g0.02", "Adaptive Mondrian")]
    xpos = np.arange(3); width = 0.24
    for j, (method, label) in enumerate(methods):
        d = final[(final.alpha == 0.10) & (final.calibration == method)]
        y = d.y.to_numpy(int); sets = d[["set_H", "set_D", "set_A"]].to_numpy(bool)
        cov = [(sets[y == c, c]).mean() for c in range(3)]
        ax.bar(xpos + (j-1)*width, cov, width, label=label)
    ax.axhline(0.90, color="#334155", linestyle="--", linewidth=1, label="90% target")
    ax.set_xticks(xpos, ["Home", "Draw", "Away"]); ax.set_ylim(0.82, 0.98)
    ax.set_ylabel("Empirical class coverage"); ax.legend(frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "fig2_class_coverage.png"), dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    targets, auto, review, abstain, auto_acc = [], [], [], [], []
    all_rows = pd.DataFrame(walk["rows"])
    for alpha in (0.30, 0.20, 0.10):
        x = all_rows[(all_rows.method == "selected_blend") & (all_rows.target_season == 2025) &
                     (all_rows.alpha == alpha) & (all_rows.calibration == "adaptive_mondrian_g0.02")].iloc[0]
        targets.append(1-alpha); auto.append(x.action_rates["automated_singleton"])
        review.append(x.action_rates["review_doubleton"]); abstain.append(x.action_rates["abstain_full_set"])
        auto_acc.append(x.singleton_accuracy)
    ax.stackplot(targets, auto, review, abstain, labels=["Automate", "Review", "Abstain"],
                 colors=["#99f6e4", "#fde68a", "#fecaca"], alpha=.9)
    ax.plot(targets, auto_acc, color="#1e3a8a", marker="o", linewidth=2, label="Automated accuracy")
    ax.set_xlabel("Target coverage"); ax.set_ylabel("Share / accuracy"); ax.set_ylim(0, 1)
    ax.set_xticks(targets, [f"{int(t*100)}%" for t in targets]); ax.legend(frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "fig3_triage_tradeoff.png"), dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for cal, label, color in (("marginal", "Marginal LAC", "#64748b"),
                              ("mondrian", "Fixed Mondrian", "#6366f1"),
                              ("adaptive_mondrian_g0.02", "Adaptive Mondrian", "#0f766e")):
        d = wf[wf.calibration == cal].groupby("target_season").coverage.mean()
        ax.plot(d.index, d.values, marker="o", linewidth=2, label=label, color=color)
    ax.axhline(.90, color="#111827", linestyle="--", linewidth=1)
    ax.set_xticks([2022,2023,2024,2025], ["22/23","23/24","24/25","25/26"])
    ax.set_ylim(.87,.94); ax.set_xlabel("Target season"); ax.set_ylabel("Empirical coverage")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig4_rolling_coverage.png"), dpi=300); plt.close(fig)

    summary = {
        "data_n": 12459, "complete_test_n": 1752,
        "selected_gamma": 0.02,
        "market_model_weight_mean": float(wf.selected_model_weight.mean()),
        "generated_tables": sorted(os.listdir(TABLES)),
        "generated_figures": sorted([x for x in os.listdir(FIGURES) if x.endswith(".png")]),
    }
    with open(os.path.join(paths.RES, "upgrade_results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
