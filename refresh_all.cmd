@echo off
REM ============================================================
REM refresh_all.cmd - One-shot pipeline refresh
REM ============================================================
REM WHAT IT DOES
REM   Rebuilds the feature matrix, reruns every experiment, and
REM   regenerates the paper tables and figures from the result
REM   files. Nothing in the paper is hand-written.
REM
REM BEFORE RUNNING
REM   1. Raw league CSVs from football-data.co.uk must be in the
REM      raw-data directory (default <repo>\data\raw; override
REM      with the FOOTBALL_DATA_DIR environment variable).
REM   2. Optional: re-crawl Understat xG if new season data exists
REM        python src\crawl_understat.py
REM
REM NOT INCLUDED
REM   src\run_llm.py is not run here: it calls a paid API, and the
REM   paper's LLM results are audited from the frozen per-match
REM   file in results\. See docs\REPRODUCE.md.
REM
REM PYTHON
REM   Uses %PY% if defined, otherwise `python` on PATH. To pin an
REM   interpreter, create a local setenv_local.cmd (git-ignored).
REM ============================================================

set SRC=%~dp0src
cd /d %~dp0

if exist setenv_local.cmd call setenv_local.cmd
if not defined PY set PY=python

echo [1/23] data_pipeline ...
%PY% %SRC%\data_pipeline.py || goto :err
echo [2/23] augment_xg ...
%PY% %SRC%\augment_xg.py || goto :err
echo [3/23] verify_no_leak ...
%PY% %SRC%\verify_no_leak.py || goto :err

echo [4/23] baselines (market/XGB/RF/Elo/Poisson) ...
%PY% %SRC%\run_baselines.py || goto :err
echo [5/23] dixon-coles ...
%PY% %SRC%\run_dixon_coles.py || goto :err
echo [6/23] walk-forward ...
%PY% %SRC%\run_walkforward.py || goto :err
echo [7/23] ablations ...
%PY% %SRC%\run_ablations.py || goto :err
echo [8/23] permutation importance ...
%PY% %SRC%\run_shap.py || goto :err
echo [9/23] by-league / by-season / LOLO ...
%PY% %SRC%\run_by_league.py || goto :err
echo [10/23] multiseed ensembles + paired bootstrap ...
%PY% %SRC%\run_multiseed_paired.py || goto :err

echo [11/23] error analysis ...
%PY% %SRC%\run_error_analysis.py || goto :err
echo [12/23] draw analysis ...
%PY% %SRC%\run_draw_analysis.py || goto :err
echo [13/23] draw diagnostics (Brier decomposition) ...
%PY% %SRC%\run_draw_deep.py || goto :err
echo [14/23] draw cost-sensitive ...
%PY% %SRC%\run_draw_cost_sensitive.py || goto :err
echo [15/23] value betting ...
%PY% %SRC%\run_value_betting.py || goto :err
echo [16/23] risk tiers / staking policies ...
%PY% %SRC%\run_risk.py || goto :err
echo [17/23] no-bet policy comparison ...
%PY% %SRC%\run_policy_comparison.py || goto :err
echo [18/23] uncertainty baselines ...
%PY% %SRC%\run_uncertainty_baselines.py || goto :err

echo [19/23] conformal sets (LAC / Mondrian / APS-RAPS diagnostics) ...
%PY% %SRC%\run_conformal_selective.py || goto :err
echo [20/23] conformal walk-forward + adaptive controller ...
%PY% %SRC%\run_conformal_walkforward.py || goto :err
echo [21/23] paired block bootstrap ...
%PY% %SRC%\analyze_conformal_paired.py || goto :err
echo [22/23] LLM audit inside triage strata (uses frozen file) ...
%PY% %SRC%\run_llm_conformal_triage.py || goto :err

echo [23/23] tables and figures ...
%PY% %SRC%\make_upgrade_outputs.py || goto :err

echo.
echo ALL DONE. Compile the paper with:
echo   cd paper ^&^& pdflatex main.tex ^&^& pdflatex main.tex
echo See docs\REPRODUCE.md for details.
exit /b 0

:err
echo.
echo FAILED at step %errorlevel% - see output above.
exit /b 1
