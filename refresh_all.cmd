@echo off
REM ============================================================
REM refresh_all.cmd - One-shot pipeline refresh
REM ============================================================
REM WHEN TO USE:
REM   1. The 2025/26 test period has completed on football-data.co.uk
REM   2. Download updated CSVs into the raw-data directory
REM      (default: <repo>\data\raw; override with the FOOTBALL_DATA_DIR
REM      environment variable, e.g. via a local setenv_local.cmd)
REM   3. (Optional) re-crawl Understat xG if new season data is available:
REM        python src\crawl_understat.py
REM   4. Run this script. It rebuilds features, reruns every experiment,
REM      and regenerates all paper tables and figures.
REM
REM Python: uses %PY% if defined, otherwise `python` on PATH. To use a
REM specific environment, create a local setenv_local.cmd (git-ignored,
REM see README) or set PY before running.
REM
REM NOTE: LLM experiments (run_llm.py) are NOT run here because they
REM       cost API credits. Run them manually if needed after refresh:
REM        python src\run_llm.py
REM       Then rerun: python src\make_tables.py
REM ============================================================

set SRC=%~dp0src
cd /d %~dp0

if exist setenv_local.cmd call setenv_local.cmd
if not defined PY set PY=python

echo [1/17] data_pipeline ...
%PY% %SRC%\data_pipeline.py || goto :err
echo [2/17] augment_xg ...
%PY% %SRC%\augment_xg.py || goto :err

echo [3/17] baselines (market/XGB/RF/Elo/Poisson) ...
%PY% %SRC%\run_baselines.py || goto :err
echo [4/17] dixon-coles ...
%PY% %SRC%\run_dixon_coles.py || goto :err
echo [5/17] ablations ...
%PY% %SRC%\run_ablations.py || goto :err
echo [6/17] permutation importance ...
%PY% %SRC%\run_shap.py || goto :err
echo [7/17] by-league / by-season / LOLO ...
%PY% %SRC%\run_by_league.py || goto :err
echo [8/17] walk-forward ...
%PY% %SRC%\run_walkforward.py || goto :err
echo [9/17] draw analysis ...
%PY% %SRC%\run_draw_analysis.py || goto :err
echo [10/17] draw cost-sensitive ...
%PY% %SRC%\run_draw_cost_sensitive.py || goto :err
echo [11/17] value betting ...
%PY% %SRC%\run_value_betting.py || goto :err
echo [12/17] risk control (SCS/UI) ...
%PY% %SRC%\run_risk.py || goto :err
echo [13/17] stop-loss sensitivity ...
%PY% %SRC%\run_stop_loss_sensitivity.py || goto :err
echo [14/17] UI weight sensitivity ...
%PY% %SRC%\run_ui_weight_sensitivity.py || goto :err
echo [15/17] error analysis ...
%PY% %SRC%\run_error_analysis.py || goto :err

echo [16/17] figures ...
%PY% %SRC%\make_figures.py || goto :err
echo [17/17] tables ...
%PY% %SRC%\make_tables.py || goto :err

echo.
echo ALL DONE. Next manual steps - see docs\REFRESH_CHECKLIST.md
exit /b 0

:err
echo.
echo FAILED at step %errorlevel% - see output above.
exit /b 1
