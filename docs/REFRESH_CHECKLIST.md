# REFRESH_CHECKLIST.md - One-shot refresh after the 2025/26 test season

> The paper states honestly: "The held-out test period (2025/26 season to
> date) is incomplete (through 2026-02-12): financial point estimates may
> shift when the season completes". This checklist is the mechanism that
> honours that promise. Every number is rebuilt automatically by
> `refresh_all.cmd` from `results/*.json`; hand-written numbers are
> forbidden (red line).

## When to run

After the 2025/26 season ends (around end of May 2026), once the
top-5-league CSVs on football-data.co.uk are complete.

## Steps

1. **Update raw data**
   - Download the latest E0/D1/F1/I1/SP1 CSVs from football-data.co.uk and
     overwrite `E:\论文\structured_data\` (keep the same directory and
     file names so the glob reader picks them up)
   - Sanity check:
     `(Get-ChildItem E:\论文\structured_data\*.csv | Get-Content | Measure-Object -Line)`,
     or simply run the next step: `data_pipeline.py` prints the
     train/val/test match counts

2. **(Optional) Update Understat xG**
   - If second-half 2025/26 xG is needed: `python src\crawl_understat.py`
     (requires network; if skipped, v3 features for new matches fall back
     to training medians and the match rate drops -- see `augment_xg.py`
     output)

3. **One-shot refresh** (double-click; ~20-40 minutes)
   ```
   refresh_all.cmd
   ```
   Runs: data_pipeline -> augment_xg -> 12 experiment scripts ->
   make_figures -> make_tables

4. **Check the numbers**
   - Inspect `results/baselines_summary.json` (market acc/logloss/ECE/ROI)
     and `results/risk_analysis.json` (UI tiers: low/medium/high acc, ROI)
   - Diff against the previous results/ (git history, or a manual diff)
   - Confirm the narrative still holds:
     - the market is not beaten (acc ~54%)
     - UI tiers stratify monotonically (low acc highest, high ROI worst)
     - the error analysis conclusions are stable (equal accuracy on
       disagreeing matches; odds movement confounded with favourites)

5. **Update the abstract wording** (manual; the only hand-edited spot)
   - `paper/sections/abstract.tex`: remove the
     "held-out test period (2025/26 season to date) is incomplete
     (through 2026-02-12)..." sentence and replace it with
     "on the complete 2025/26 test season (N matches)", where N comes
     from `results/baselines_summary.json`
   - Grep for "incomplete / through 2026-02-12" in intro/conclusion and
     update accordingly

6. **(Optional) Re-run the LLM experiments**
   - Full test set (1104x3 calls, ~$0.55): `python src\run_llm.py`
   - Afterwards re-run `python src\make_tables.py` to refresh the LLM tables
   - If the test-period size changed, the LLM run cost changes accordingly

7. **Recompile the paper**
   - Overleaf: re-upload paper_overleaf.zip (or compile locally)
   - Verify every table number matches results/*.json (make_tables
     guarantees this)

8. **Sync the public repository**
   - `git add -A && git commit -m "refresh test season 2025/26" && git tag v1.1.0`
   - Enable the proxy before pushing (http://127.0.0.1:10090); for large
     packs add `git config http.postBuffer 524288000`

## Red lines

- Every number must be produced by a script; `refresh_all.cmd` enforces
  the full chain
- Report results as they are, not the best case: if after a refresh the
  market baseline is stronger or the UI stratification is weaker, write
  that honestly; do not tune thresholds to preserve the narrative
- API key files (llm-config.local.json / embed-config.local.json) never
  enter the repository (.gitignore excludes them)
