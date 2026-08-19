# Football Outcome Forecasting: Leakage-Controlled Benchmark and Uncertainty-Driven Risk Control

Code and data pipelines for the paper:

> **Multi-Source Pre-Match Signal Fusion and Uncertainty-Driven Risk Control for
> Football Outcome Forecasting: A Leakage-Controlled Evaluation of Machine
> Learning and Large Language Models**

This repository reproduces every number in the paper. All figures are computed
by the scripts below from the raw data; no result in the paper is hand-written.

## Highlights

- 11,811 matches, top-5 European leagues, 2019/20–2025/26 (test = 2025/26,
  incomplete through 2026-02-12, 1,104 matches).
- Strictly chronological train/val/test split; every feature is pre-match;
  scalers and missing-value statistics are fit on the training split only.
- Baselines: de-vigged closing odds (market), Poisson, Dixon–Coles, Elo,
  Random Forest, XGBoost, domain-enhanced LLM (DeepSeek, optional local Qwen).
- Risk control: corrected Scenario Complexity Score (SCS) and Uncertainty
  Index (UI) with tiered staking, no-bet policy, and stop-loss.
- All financial results under a fully specified equal-stake protocol with
  bootstrap 95% confidence intervals.

## Repository layout

```
src/
  data_pipeline.py          # leakage-free feature pipeline (v3, incl. xG)
  augment_xg.py             # joins Understat xG features into the pipeline
  crawl_understat.py        # fetches Understat xG data (needs network)
  verify_no_leak.py         # manual cross-check that features are pre-match
  run_baselines.py          # market / XGB / RF / Elo / Poisson comparison
  run_dixon_coles.py        # Dixon–Coles baseline (vectorized MLE)
  run_ablations.py          # feature-group ablations
  run_shap.py               # permutation importance
  run_by_league.py          # per-league, per-season, leave-one-league-out
  run_walkforward.py        # walk-forward evaluation across seasons
  run_draw_analysis.py      # draw class metrics, confusion matrix, threshold scan
  run_draw_cost_sensitive.py# cost-sensitive draw training
  run_value_betting.py      # EV-threshold betting, opening-vs-closing, Kelly
  run_risk.py               # SCS/UI tiers, strategy comparison, threshold sensitivity
  run_ui_weight_sensitivity.py  # UI weight robustness (val)
  run_stop_loss_sensitivity.py  # stop-loss parameter robustness (test)
  run_llm.py                # LLM forecasting (DeepSeek / local Qwen), resumable
  analyze_llm.py            # LLM post-analysis (UI tiers, consistency)
  analyze_llm_compare.py    # temperature sensitivity + open-vs-closed LLM
  risk.py                   # SCS/UI/risk simulation implementations
  evaluate.py               # metrics + bootstrap CI + betting simulation
  make_tables.py            # generates all paper tables from results/*.json
  make_figures.py           # generates all paper figures from real data
  llm/llm_client.py         # pluggable LLM client (reads local config only)
  llm/prompts.py            # domain-enhanced prompt templates
results/                   # all experiment outputs (JSON) = paper data source
paper/                     # LaTeX source, tables, figures, compiled PDF
data/processed/all_matches_featurized.csv  # final feature matrix (optional download)
```

## Setup

Python 3.13 (3.10+ should work). Dependencies in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Optional LLM experiments need an API key. Copy the template and fill it in
(the file is never read into the repository):

```bash
cp src/llm/llm-config.local.json.example src/llm/llm-config.local.json
```

## Data

1. **Structured match data** (football-data.co.uk format, leagues E0/D1/F1/I1/SP1,
   seasons 2019/20–2025/26): download the CSV files into `data/raw/` (paths are
   configured at the top of `src/data_pipeline.py`). The files are freely
   available from football-data.co.uk.
2. **xG data** (optional but used in the paper): fetch from Understat:

```bash
python src/crawl_understat.py   # writes data/raw_understat/*.csv (35 league-seasons)
```

If you use the provided `data/processed/all_matches_featurized.csv`, steps 1–2
are only needed if you want to rebuild features from scratch.

## Reproduction (end-to-end)

```bash
# 1. Build the feature matrix and splits (no leakage)
python src/data_pipeline.py            # v2 (structured + market + referee)
python src/augment_xg.py               # v3 (adds xG features; overwrites processed/)

# 2. Verify the pipeline has no leakage (manual recomputation checks)
python src/verify_no_leak.py

# 3. Baselines and analyses (each writes results/*.json)
python src/run_baselines.py
python src/run_dixon_coles.py
python src/run_ablations.py
python src/run_shap.py
python src/run_by_league.py
python src/run_walkforward.py
python src/run_draw_analysis.py
python src/run_draw_cost_sensitive.py
python src/run_value_betting.py
python src/run_risk.py
python src/run_ui_weight_sensitivity.py
python src/run_stop_loss_sensitivity.py

# 4. LLM experiments (requires config; ~$0.55 for the full test season with DeepSeek)
python src/run_llm.py --provider deepseek --n_matches 1104 --n_samples 3
python src/analyze_llm.py
python src/analyze_llm_compare.py

# 5. Paper artifacts (tables and figures are generated, never hand-written)
python src/make_tables.py
python src/make_figures.py
```

## How results map to the paper

| Paper content | Result file | Produced by |
|---|---|---|
| Table 1 (main comparison) | `results/baselines_summary.json`, `results/dixon_coles.json`, `results/llm_analysis.json` | `run_baselines.py` etc. |
| Table 2 (ablations) | `results/ablations_summary.json` | `run_ablations.py` |
| Table 3 (feature importance) | `results/feature_importance.json` | `run_shap.py` |
| Tables 4–5 (leagues, LOLO) | `results/by_league.json` | `run_by_league.py` |
| Table 6 (walk-forward) | `results/walkforward.json` | `run_walkforward.py` |
| Tables 7–8 (draws) | `results/draw_analysis.json`, `results/draw_cost_sensitive.json` | `run_draw_analysis.py` |
| Tables 9–10 (value betting) | `results/value_betting.json` | `run_value_betting.py` |
| Tables 11–14 (SCS/UI/strategies) | `results/risk_analysis.json` | `run_risk.py` |
| Tables 15–16 (LLM) | `results/llm_compare.json`, `results/llm_analysis.json` | `run_llm.py`, `analyze_llm*.py` |
| Figures 1–5 | `paper/figures/fig*.png` | `make_figures.py` |

## LLM configuration

`src/llm/llm-config.local.json` (user-created, git-ignored):

```json
{
  "default_provider": "deepseek",
  "deepseek": { "api_key": "YOUR_KEY", "base_url": "https://api.deepseek.com", "model": "deepseek-chat" },
  "local":    { "base_url": "http://127.0.0.1:8001", "model": "qwen2.5-coder-7b" }
}
```

The client reads this file at runtime only and never prints the key.

## License

MIT — see [LICENSE](LICENSE). Data files retain their original licenses
(football-data.co.uk, Understat).

## Citation

```
[to be added upon publication]
```
