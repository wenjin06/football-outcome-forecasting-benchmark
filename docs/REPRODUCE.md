# Reproduction guide

Everything reported in the paper is produced by the scripts in `src/` from
public raw data. No number or figure in the paper is hand-written.

## What this repository contains

| Path | Role |
|---|---|
| `src/` | data construction, experiments, conformal triage, artifact generation |
| `results/` | experiment outputs; every table and figure is generated from these |
| `paper/` | manuscript source (`main.tex`), `tables/`, `figures/` |
| `data/processed/all_matches_featurized.csv` | final feature matrix (provided so the reported numbers can be reproduced without downloading raw data) |

Frozen inputs that are reused rather than regenerated:

- `results/llm_deepseek_t0.3_per_match.csv` — per-match LLM probabilities for the
  1,104-match subset. Re-running the LLM experiment requires an API key and
  costs money, so the frozen file is the reference for the audit in Table 6.

## 1. Environment

```bash
pip install -r requirements.txt
```

Paths are resolved by `src/paths.py` relative to the repository root, so no
editing is required. Three optional environment variables override the
defaults:

| Variable | Default | Purpose |
|---|---|---|
| `FOOTBALL_DATA_DIR` | `data/raw/` | raw football-data.co.uk CSVs |
| `FOOTBALL_PROCESSED_DIR` | `data/processed/` | feature matrices and splits |
| `FOOTBALL_RESULTS_DIR` | `results/` | experiment outputs |

Setting them is useful for running an experiment without overwriting the
frozen artefacts reported in the paper.

## 2. Raw data

Two public sources are used:

1. **football-data.co.uk** — league CSVs for E0, D1, F1, I1, SP1, seasons
   2019/20–2025/26. Download into `FOOTBALL_DATA_DIR` (default `data/raw/`).
   Filenames must keep the standard `<league><season>.csv` form (for example
   `E0.csv`, `E0_2021.csv`), because the loader globs the directory.
2. **Understat** — expected goals, pressing and deep-progression statistics:

```bash
python src/crawl_understat.py
```

Step 2 needs network access. If you use the provided feature matrix in
`data/processed/`, both steps are only needed to rebuild features from scratch.

## 3. Full refresh

Two options:

```bash
# Windows one-shot
refresh_all.cmd

# or run the stages manually, in this order
python src/data_pipeline.py
python src/augment_xg.py
python src/verify_no_leak.py
python src/run_baselines.py
python src/run_dixon_coles.py
python src/run_walkforward.py
python src/run_ablations.py
python src/run_shap.py
python src/run_by_league.py
python src/run_multiseed_paired.py
python src/run_error_analysis.py
python src/run_draw_analysis.py
python src/run_draw_deep.py
python src/run_draw_cost_sensitive.py
python src/run_value_betting.py
python src/run_risk.py
python src/run_policy_comparison.py
python src/run_uncertainty_baselines.py
python src/run_conformal_selective.py
python src/run_conformal_walkforward.py
python src/analyze_conformal_paired.py
python src/run_llm_conformal_triage.py
python src/make_upgrade_outputs.py
```

`make_upgrade_outputs.py` regenerates `paper/tables/*.tex` and
`paper/figures/*.png` from the result files. The LLM experiment itself
(`src/run_llm.py`) is deliberately excluded from the refresh: it calls a paid
API, and the paper's LLM results are audited from the frozen per-match file.

## 4. Verifying a refresh

```bash
# 1. table and figure inputs are complete
python src/make_upgrade_outputs.py

# 2. the manuscript compiles from the repository alone
cd paper && pdflatex main.tex && pdflatex main.tex
```

Expected: `main.pdf` compiles with no unresolved references and no overfull
boxes.

## 5. Notes on determinism

The pipeline is deterministic: two consecutive runs of the same code reproduce
all 1,433 numeric fields of `results/conformal_selective_prototype.json`
exactly. Editing a script can legitimately change results, so if you compare
against the published numbers, check out the commit that produced the paper
first (`git log --oneline`).

## 6. Previous versions

Two annotated tags exist for traceability:

| Tag | Points at | Contents |
|---|---|---|
| `archive/previous-manuscript` | `b9d0eeb` | the previous manuscript (uncertainty-index staking study) and its supporting scripts |
| `archive/pre-cleanup` | `0f06657` | the state before legacy material was removed from `main` |

To inspect or restore an earlier state:

```bash
git checkout archive/previous-manuscript     # detached checkout
git checkout archive/previous-manuscript -- paper/   # restore only the old paper
```
