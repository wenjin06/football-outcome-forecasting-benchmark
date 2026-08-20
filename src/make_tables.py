"""
论文表格生成器：从 results/*.json 自动生成 LaTeX 表格
====================
所有论文数字必须经由此脚本产出，禁止手写。
输出：paper/tables/tableN.tex（experiments.tex 用 \\input 引用）

用法：python src/make_tables.py
"""
import os
import json

RES = r"E:\论文\sci_redo\results"
OUT = r"E:\论文\sci_redo\paper\tables"
os.makedirs(OUT, exist_ok=True)


def load(name):
    with open(os.path.join(RES, name), "r", encoding="utf-8") as f:
        return json.load(f)


def pct(x, digits=1):
    return f"{x*100:.{digits}f}\\%" if x is not None else "---"


def sanitize_label(s):
    """把表格文本中的 < > 等字符转成 LaTeX 数学模式，避免 T1 编码渲染成 ¡/¿。"""
    return (s.replace("<=", "$\\leq$").replace(">=", "$\\geq$")
             .replace("<", "$<$").replace(">", "$>$"))


def num(x, digits=3):
    return f"{x:.{digits}f}" if x is not None else "---"


def ci(x):
    if not x:
        return "---"
    return f"[{x[0]*100:.1f},{x[1]*100:.1f}]"


def table_wrap(caption, label, header, rows, span=False):
    env = "table*" if span else "table"
    width = "\\textwidth" if span else "\\linewidth"
    spec = "[tbp]" if span else "[htbp]"
    lines = [f"\\begin{{{env}}}{spec}", "\centering", "\small",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\resizebox{{{width}}}{{!}}{{",
             "\\begin{tabular}{" + "l" + "c" * (len(header) - 1) + "}",
             "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(sanitize_label(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", f"\\end{{{env}}}"]
    return "\n".join(lines) + "\n"


def save(name, content):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  {name}")


print("生成表格 -> paper/tables/")

# ============ 表1：主对比 ============
bs = load("baselines_summary.json")
dc = load("dixon_coles.json")
la = load("llm_analysis.json")

rows = []
order = [("market", "Market (de-vig)"), ("dixon_coles", "Dixon--Coles"),
         ("poisson", "Poisson"), ("elo", "Elo"), ("rf", "Random Forest"),
         ("xgb", "XGBoost"), ("llm", "LLM (DeepSeek)")]
for key, name in order:
    if key == "dixon_coles":
        d = dc["overall"]
        rows.append([name, pct(d["acc"]), num(d["logloss"]), num(d.get("brier")),
                     num(d.get("ece")), pct(d["roi"]), pct(d.get("mdd")), num(d.get("sharpe"))])
    elif key == "llm":
        d = la["main_row_llm"]
        rows.append([name, pct(d["acc"]), num(d["logloss"]), num(d["brier"]),
                     num(d["ece"]), pct(d["roi"]), pct(d["mdd"]), num(d["sharpe"])])
    else:
        d = bs[key]
        rows.append([name, pct(d["accuracy"]), num(d["log_loss"]), num(d["brier"]),
                     num(d["ece"]), pct(d["roi_all"]), pct(d["mdd_all"]), num(d["sharpe_all"])])
save("table1_main.tex", table_wrap(
    "Overall performance on the held-out test season (1,104 matches). "
    "The LLM parsed 100\\% of matches at a cost of \\$0.55. Bootstrap 95\\% "
    "confidence intervals are reported in the text.", "tab:main",
    ["Model", "Acc", "LogLoss", "Brier", "ECE", "ROI", "MDD", "Sharpe"], rows,
    span=True))

# ============ 表2：消融 ============
ab = load("ablations_summary.json")
order2 = [("full", "Full (60 features)"), ("no_odds", "w/o market odds"),
          ("no_referee", "w/o referee"), ("no_rank", "w/o rank"),
          ("no_xg", "w/o xG"), ("structured_only", "Structured only"),
          ("no_form", "w/o form")]
rows = []
for key, name in order2:
    d = ab[key]
    rows.append([name, str(d["n_features"]), pct(d["accuracy"]),
                 num(d["log_loss"]), num(d["ece"]), pct(d["roi_all"]),
                 pct(d["roi_t04"])])
save("table2_ablation.tex", table_wrap(
    "Ablation study (XGBoost, test set). ROI-all: all bets; ROI-t0.4: "
    "bets with model probability at least 0.4.", "tab:ablation",
    ["Configuration", "\\#Feat", "Acc", "LogLoss", "ECE", "ROI(all)", "ROI(t$\\geq$0.4)"], rows,
    span=True))

# ============ 表3：特征重要性分组 ============
fi = load("feature_importance.json")
g = fi["group_importance"]
tot = sum(g.values())
rows = []
for k, v in sorted(g.items(), key=lambda x: -x[1]):
    display = {"market": "Market", "xg": "xG", "season_cum": "Season cumulative",
               "rank": "League rank", "referee": "Referee",
               "form_rolling": "Rolling form"}.get(k, k)
    rows.append([display, num(v, 4), pct(v / tot, 1)])
save("table3_importance.tex", table_wrap(
    "Permutation importance by feature group (mean increase in log loss "
    "over 5 permutations, test set).", "tab:importance",
    ["Group", "Importance", "Share"], rows))

# ============ 表4：分联赛 ============
bl = load("by_league.json")
rows = []
for div, d in bl["by_league_test"].items():
    rows.append([div, str(d["n"]), pct(d["accuracy"]), num(d["log_loss"]),
                 pct(d["roi"]), pct(d["mdd"])])
save("table4_byleague.tex", table_wrap(
    "Per-league performance of the global XGBoost model on the test set.",
    "tab:byleague", ["League", "N", "Acc", "LogLoss", "ROI", "MDD"], rows))

# ============ 表5：LOLO ============
rows = []
for div, d in bl["leave_one_league_out"].items():
    rows.append([div, str(d["n"]), pct(d["accuracy"]), num(d["log_loss"]),
                 pct(d["roi"])])
save("table5_lolo.tex", table_wrap(
    "Leave-one-league-out generalization: model trained on the other four "
    "leagues, evaluated on the held-out league (test set).", "tab:lolo",
    ["Held-out league", "N", "Acc", "LogLoss", "ROI"], rows))

# ============ 表6：walk-forward ============
wf = load("walkforward.json")
rows = []
for season, d in wf.items():
    m = d["market"]; xe = d["xgb_expanding"]
    xr = d.get("xgb_rolling2")
    rows.append([season, str(d["n"]), pct(m["acc"]), pct(xe["acc"]),
                 num(xe["logloss"]),
                 pct(xr["acc"]) if xr else "---"])
save("table6_walkforward.tex", table_wrap(
    "Walk-forward evaluation across four test seasons. Market: de-vigged "
    "closing odds. XGB-exp: expanding training window. XGB-roll2: last two "
    "seasons only.", "tab:walkforward",
    ["Season", "N", "Market", "XGB-exp", "LL(exp)", "XGB-roll2"], rows))

# ============ 表7：平局类别 ============
da = load("draw_analysis.json")
x = da["xgb"]
rows = []
for i, cls in enumerate(["Home win", "Draw", "Away win"]):
    rows.append([cls, num(x["class_precision"][i]), num(x["class_recall"][i]),
                 num(x["class_f1"][i]), num(x["class_ece"][str(i)])])
rows.append(["Pred. draw rate", pct(da["xgb"]["pred_draw_rate"]), "---", "---", "---"])
rows.append(["Empirical draw rate", pct(da["empirical_draw_rate"]), "---", "---", "---"])
save("table7_draw_class.tex", table_wrap(
    "Class-wise metrics (XGBoost, test set). Empirical draw rate is 25.5\\% "
    "but the model predicts a draw in only 2.3\\% of matches.",
    "tab:drawclass", ["Class", "Prec", "Recall", "F1", "ECE"], rows))

# ============ 表8：平局重加权 ============
dcs = load("draw_cost_sensitive.json")
rows = []
for w in ["1.0", "1.5", "2.0", "2.5", "3.0"]:
    d = dcs[w]
    rows.append([w, pct(d["acc"]), num(d["macro_f1"]), num(d["draw_recall"]),
                 num(d["draw_precision"]), num(d["logloss"])])
save("table8_draw_weight.tex", table_wrap(
    "Cost-sensitive training: draw class weight sweep (XGBoost, test set). "
    "Weight 1.0 is the baseline.", "tab:drawweight",
    ["w(draw)", "Acc", "Macro-F1", "Draw R", "Draw P", "LogLoss"], rows))

# ============ 表9：价值投注 ============
vb = load("value_betting.json")
rows = []
for e in vb["value_ev_scan"]:
    rows.append([num(e["threshold"], 2), pct(e["roi"]), num(e["sharpe"]),
                 pct(e["mdd"]), str(e["n_bets"]), pct(e["coverage"])])
save("table9_value.tex", table_wrap(
    "Value betting: bet when EV = $p \\times odds - 1$ exceeds the threshold "
    "(XGBoost probabilities, closing odds). Higher thresholds select "
    "increasingly confident bets and lose more money.",
    "tab:value", ["EV thresh.", "ROI", "Sharpe", "MDD", "N bets", "Coverage"], rows))

# ============ 表10：校准后价值投注 ============
rows = []
for e in vb["value_ev_scan_calibrated"]:
    rows.append([num(e["threshold"], 2), pct(e["roi"]), str(e["n_bets"]),
                 pct(e["coverage"])])
save("table10_value_cal.tex", table_wrap(
    "Value betting after isotonic calibration (fit on validation). "
    "Losses shrink but remain negative: calibration cannot create "
    "information the model does not have.", "tab:valuecal",
    ["EV thresh.", "ROI", "N bets", "Coverage"], rows))

# ============ 表11：SCS 分层 ============
ra = load("risk_analysis.json")
rows = []
for label, d in ra["scs_tiers"].items():
    rows.append([label, str(d["n"]), pct(d["accuracy"]), pct(d["roi"])])
save("table11_scs.tex", table_wrap(
    "Scenario-complexity tiers (XGBoost, test set, thresholds 1.4/1.8). "
    "Higher SCS indicates harder matches.", "tab:scs",
    ["SCS tier", "N", "Acc", "ROI"], rows))

# ============ 表12：UI 分层 XGB ============
rows = []
for label, d in ra["ui_tiers"].items():
    if d.get("n", 0) == 0:
        continue
    rows.append([label, str(d["n"]), pct(d["accuracy"]), pct(d["roi"]),
                 pct(d["mdd"])])
save("table12_ui_xgb.tex", table_wrap(
    "Uncertainty-index tiers (XGBoost, test set, thresholds 0.30/0.45). "
    "High-uncertainty matches are excluded by the no-bet policy; their "
    "predictive accuracy is reported separately.", "tab:uixgb",
    ["UI tier", "N", "Acc", "ROI", "MDD"], rows))

# ============ 表13：UI 分层 LLM ============
rows = []
for label, d in la["ui_tiers"].items():
    if d.get("n", 0) == 0:
        continue
    rows.append([label, str(d["n"]), pct(d["accuracy"]), pct(d["roi"]),
                 pct(d["mdd"])])
save("table13_ui_llm.tex", table_wrap(
    "Uncertainty-index tiers for the LLM (test set). The same stratification "
    "pattern replicates across model families.", "tab:uillm",
    ["UI tier", "N", "Acc", "ROI", "MDD"], rows))

# ============ 表14：策略对比 ============
rows = []
for key, name in [("naive", "Naive (all bets)"), ("t04", "Conf. threshold 0.4"),
                  ("ui_tier", "UI tiers + no-bet"), ("ui_tier_sl", "UI tiers + stop-loss")]:
    d = ra["strategies"][key]
    rows.append([name, pct(d["roi"]), num(d["sharpe"]), pct(d["mdd"]),
                 str(d["n_bets"]),
                 ci(d.get("roi_ci"))])
save("table14_strategy.tex", table_wrap(
    "Betting-strategy comparison (XGBoost probabilities, test set, bootstrap "
    "95\\% CI for ROI). No-bet matches are excluded from ROI but their "
    "coverage is reported.", "tab:strategy",
    ["Strategy", "ROI", "Sharpe", "MDD", "N bets", "ROI 95\\% CI"], rows,
    span=True))

# ============ 表15：LLM 温度与开源对照 ============
lc = load("llm_compare.json")
rows = []
for k, name in [("t0.3", "DeepSeek t=0.3"), ("t0.7", "DeepSeek t=0.7")]:
    d = lc["temperature"][k]
    c = lc["temperature"][f"{k}_consistency"]
    rows.append([name, pct(d["acc"]), num(d["logloss"]), num(d["ece"]),
                 num(c["cons_mean"], 3), num(c["point_biserial"])])
save("table15_temp.tex", table_wrap(
    "Temperature sensitivity (same 120 matches). Multi-sample consistency "
    "remains near 1.0 at both temperatures and does not correlate with "
    "correctness.", "tab:temp",
    ["Config", "Acc", "LogLoss", "ECE", "Cons.", "r(cons,corr)"], rows))

rows = []
rn = lc.get("reasoner_notes", {})
if rn:
    # 同批前 120 场三模型对比（reasoner 为单采样，其余为 3 采样平均）
    items = [("deepseek_t0.3_same_120", "DeepSeek-Chat (API)"),
             ("deepseek_reasoner", "DeepSeek-Reasoner (API)"),
             ("qwen_local_same_120", "Qwen2.5-Coder-7B (local)")]
    for k, name in items:
        d = rn.get(k) or lc["open_vs_closed"].get(k)
        if not d:
            continue
        rows.append([name, pct(d["acc"]), num(d["logloss"]), num(d["ece"]),
                     pct(d["roi"]), pct(d["win_rate"])])
    save("table16_open.tex", table_wrap(
        "LLM model comparison on the same first 120 test-set matches: "
        "DeepSeek-Chat and the local Qwen with 3 samples each, "
        "DeepSeek-Reasoner with 1 sample (reasoning models are ~10x slower; "
        "runtime constraint). This comparison is robustness evidence only: "
        "the sample is small and confidence intervals overlap, so no "
        "superiority claim is made for any model.",
        "tab:open", ["Model", "Acc", "LogLoss", "ECE", "ROI", "Win rate"], rows))
else:
    rows = []
    items = [("deepseek_t0.3", "DeepSeek-Chat (API)"),
             ("qwen_local", "Qwen2.5-Coder-7B (local)")]
    for k, name in items:
        d = lc["open_vs_closed"][k]
        rows.append([name, pct(d["acc"]), num(d["logloss"]), num(d["ece"]),
                     pct(d["roi"]), pct(d["win_rate"])])
    save("table16_open.tex", table_wrap(
        "LLM model comparison on the same 200 test-set matches. This "
        "comparison is robustness evidence only: the sample is small and "
        "confidence intervals overlap, so no superiority claim is made for "
        "any model.",
        "tab:open", ["Model", "Acc", "LogLoss", "ECE", "ROI", "Win rate"], rows))

# ============ 表17：错误分析（失败与异动/分歧关联） ============
ea = load("error_analysis.json")
mvm = ea["market_vs_model"]
rows = [
    ["All matches", str(mvm["n_agree"] + mvm["n_disagree"]),
     pct(mvm["acc_model_overall"]), pct(mvm["acc_market_overall"])],
    ["Model = market", str(mvm["n_agree"]),
     pct(mvm["acc_model_on_agree"]), pct(mvm["acc_market_on_agree"])],
    ["Model $\\neq$ market", str(mvm["n_disagree"]),
     pct(mvm["acc_model_on_disagree"]), pct(mvm["acc_market_on_disagree"])],
]
for thr in ["50", "60", "70"]:
    d = ea[f"high_conf_{thr}"]
    rows.append([f"conf $\\geq$ 0.{thr}", f"{d['model_n']}/{d['market_n']}",
                 pct(d["model_error_rate"]), pct(d["market_error_rate"])])

rows += [
    ["Odds-move Q0 (small)", str(ea["odds_movement_bins"][0]["n"]),
     pct(ea["odds_movement_bins"][0]["error_rate"]), pct(ea["odds_movement_bins"][0]["fav_share"], 0)],
    ["Odds-move Q4 (large)", str(ea["odds_movement_bins"][4]["n"]),
     pct(ea["odds_movement_bins"][4]["error_rate"]), pct(ea["odds_movement_bins"][4]["fav_share"], 0)],
    ["Dispersion Q0 (low)", str(ea["market_dispersion_bins"][0]["n"]),
     pct(ea["market_dispersion_bins"][0]["error_rate"]), pct(ea["market_dispersion_bins"][0]["fav_share"], 0)],
    ["Dispersion Q4 (high)", str(ea["market_dispersion_bins"][4]["n"]),
     pct(ea["market_dispersion_bins"][4]["error_rate"]), pct(ea["market_dispersion_bins"][4]["fav_share"], 0)],
]
de = ea["draw_errors"]
rows += [
    ["Draws, correctly predicted", str(de["correctly_predicted"]["n"]),
     num(de["correctly_predicted"]["mean_conf"], 3), "---"],
    ["Draws, misclassified", str(de["misclassified"]["n"]),
     num(de["misclassified"]["mean_conf"], 3), "---"],
]
lines = ["\\begin{table*}[tbp]", "\\centering", "\\small",
         "\\caption{Error analysis on the test set. Rows 1--3: accuracy of XGBoost",
         "vs de-vigged closing odds on subsets where the two agree or disagree.",
         "Rows 4--6: error rate among predictions with model confidence at least",
         "0.50/0.60/0.70 (N = model/market). Rows 7--10: error rate and share of",
         "strong favourites (min opening odds $\\leq$ 1.6) for the extreme",
         "quintiles of odds movement and of market dispersion ((Max-Avg)/Avg",
         "closing odds). Rows 11--12: mean model confidence for draws that were",
         "correctly vs incorrectly classified.}", "\\label{tab:error}",
         "\\resizebox{\\textwidth}{!}{",
         "\\begin{tabular}{lccc}", "\\toprule",
         "Group & N & Acc/Err. & Fav. share \\\\", "\\midrule"]
for r in rows:
    lines.append(" & ".join(sanitize_label(c) for c in r) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"]
save("table17_error.tex", "\n".join(lines) + "\n")

# ============ 表18：难度信号分层（市场概率 vs UI） ============
dm = load("ui_vs_market_difficulty.json")
rows = []
for b, d in enumerate(dm["mkt_max_quintiles"]):
    rows.append([f"Q{b}", str(d["n"]), pct(d["acc"]), pct(d["fav_share"], 0)])
rows.append(["UI low (0.30)", str(dm["ui_tiers"]["low"]["n"]),
             pct(dm["ui_tiers"]["low"]["acc"]), pct(dm["ui_tiers"]["low"]["fav_share"], 0)])
rows.append(["UI high (no-bet)", str(dm["ui_tiers"]["high"]["n"]),
             pct(dm["ui_tiers"]["high"]["acc"]), pct(dm["ui_tiers"]["high"]["fav_share"], 0)])
save("table18_difficulty.tex", table_wrap(
    "Difficulty signals on the test set. First five rows: quintiles of the "
    "de-vigged closing-odds max probability (market strength). Last two rows: "
    "the UI tiers of Table~\\ref{tab:uixgb}. Fav. share = fraction of matches "
    "with minimum opening odds $\\leq$ 1.6. The market's own probability "
    "level is itself a strong difficulty signal, and the UI adds to it (see "
    "text).", "tab:difficulty",
    ["Group", "N", "Acc", "Fav. share"], rows))

# 交叉增量（文字引用数字，脚本产出）
cross = {f"{c['mkt_q']}_{c['ui_q']}": c for c in dm["mkt_x_ui_cross"]}

# ============ 表19：策略级对比（UI vs market-conf vs SCS vs model-conf vs random） ============
pc = load("policy_comparison.json")
rows = []
for name in ["UI", "SCS", "market_conf", "model_conf", "random"]:
    for r in pc["strategies"][name]:
        if r["coverage"] not in (0.7, 0.5):
            continue
        label = {"UI": "UI", "SCS": "SCS", "market_conf": "Market conf.",
                 "model_conf": "Model conf.", "random": "Random"}[name]
        rows.append([label, f"{r['coverage']:.1f}", pct(r["acc"]), pct(r["roi"]),
                     f"[{r['roi_ci'][0]*100:.1f},{r['roi_ci'][1]*100:.1f}]",
                     num(r["sharpe"]), num(r["avg_odds"], 2)])
save("table19_policy.tex", table_wrap(
    "No-bet policy comparison at two coverage levels (test set, XGBoost "
    "probabilities, equal-stake betting). Policies drop the highest-risk "
    "matches according to each score; coverage 0.5 drops half the matches. "
    "Market confidence is the strongest accuracy stratifier but selects "
    "low-odds favourites (avg odds 1.43 at coverage 0.7), so its financial "
    "stratification fails (negative ROI at coverage 0.5); uncertainty-based "
    "policies (UI, SCS) keep positive ROI. Bootstrap 95\\% CIs for ROI all "
    "cross zero: gains are not statistically significant in a single season.",
    "tab:policy", ["Policy", "Cov", "Acc", "ROI", "ROI 95\\% CI", "Sharpe", "Avg odds"],
    rows))

print("\n全部表格已生成:", len(os.listdir(OUT)), "个文件")
