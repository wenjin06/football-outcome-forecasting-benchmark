"""
论文图表生成器：从 results/*.json 与管道产物生成论文配图
====================
输出：paper/figures/fig1_framework.png ~ fig5_confusion.png（300dpi）
图表全部由真实数据绘制，禁止手工美化数字。
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from xgboost import XGBClassifier

RES = r"E:\论文\sci_redo\results"
OUT = r"E:\论文\sci_redo\paper\figures"
DATA = r"E:\论文\sci_redo\data\processed"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 300, "savefig.dpi": 300})

feat = pd.read_csv(os.path.join(DATA, "all_matches_featurized.csv"), parse_dates=["Date"])
drop_cols = ["Div", "Date", "Season", "HomeTeam", "AwayTeam", "FTR", "y"]
feature_cols = [c for c in feat.columns if c not in drop_cols and feat[c].notna().sum() > 0]
medians = feat[feat["Date"] < "2024-08-01"][feature_cols].median()
feat[feature_cols] = feat[feature_cols].fillna(medians)
train = feat[feat["Date"] < "2024-08-01"]
val = feat[(feat["Date"] >= "2024-08-01") & (feat["Date"] < "2025-08-01")]
test = feat[feat["Date"] >= "2025-08-01"]
yte = test["y"].values.astype(int)

print("[1/5] 训练 XGB（用于可靠性图/ROI 曲线）...")
xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                    early_stopping_rounds=30, random_state=42)
xgb.fit(train[feature_cols], train["y"].values.astype(int),
        eval_set=[(val[feature_cols], val["y"].values.astype(int))], verbose=False)
proba_xgb = xgb.predict_proba(test[feature_cols])

# ============ 图1：框架图 ============
print("[2/5] 框架图 ...")
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.axis("off")


def box(x, y, w, h, text, fc="#eef3fb", ec="#4a6fa5"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       fc=fc, ec=ec, lw=1.2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.2, color="#333333"))


box(0.02, 0.35, 0.16, 0.5, "Structured\nstats")
box(0.02, 0.75, 0.16, 0.5, "xG time series")
box(0.02, -0.05, 0.16, 0.5, "Market odds\n(8 books)")
box(0.02, -0.45, 0.16, 0.5, "Referee\nhistory")
box(0.26, 0.10, 0.18, 0.7, "Pre-match feature\ncard (60 feats)\n[no leakage]")
box(0.52, 0.10, 0.18, 0.7, "Forecasters:\nmarket / Poisson /\nDC / Elo / RF /\nXGB / LLM")
box(0.78, 0.10, 0.20, 0.7, "SCS + UI\nrisk tiers\nno-bet / stake")
for x in [0.02, 0.02, 0.02, 0.02]:
    arrow(x + 0.16, 0.45, 0.26, 0.45)
arrow(0.44, 0.45, 0.52, 0.45)
arrow(0.70, 0.45, 0.78, 0.45)
ax.set_xlim(0, 1.02)
ax.set_ylim(-0.6, 1.35)
ax.text(0.02, 1.25, "Pre-match data", fontsize=9, fontweight="bold")
ax.text(0.26, 1.25, "Features", fontsize=9, fontweight="bold")
ax.text(0.52, 1.25, "Models", fontsize=9, fontweight="bold")
ax.text(0.78, 1.25, "Risk control", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_framework.png"), bbox_inches="tight")
plt.close()

# ============ 图2：可靠性图 ============
print("[3/5] 可靠性图 ...")
llm = pd.read_csv(os.path.join(RES, "llm_deepseek_t0.3_per_match.csv"))
llm = llm.dropna(subset=["p_H"]).reset_index(drop=True)
proba_llm = llm[["p_H", "p_D", "p_A"]].values
y_llm = llm["y"].values.astype(int)
mkt = test[["mkt_prob_H", "mkt_prob_D", "mkt_prob_A"]].values

fig, ax = plt.subplots(figsize=(3.6, 3.2))
for name, proba, y in [("Market", mkt, yte), ("XGBoost", proba_xgb, yte),
                       ("LLM", proba_llm, y_llm)]:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    acc = (pred == y).astype(float)
    xs, ys = [], []
    for i in range(10):
        mask = (conf > i / 10) & (conf <= (i + 1) / 10)
        if mask.sum() >= 30:
            xs.append(conf[mask].mean())
            ys.append(acc[mask].mean())
    ax.plot(xs, ys, marker="o", ms=3, label=name)
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
ax.set_xlabel("Mean predicted confidence")
ax.set_ylabel("Observed accuracy")
ax.legend(fontsize=7, frameon=False)
ax.set_title("Reliability (binned max-probability)", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_reliability.png"))
plt.close()

# ============ 图3：UI 分布 + 分层 ============
print("[4/5] UI 分布与分层 ...")
from risk import compute_ui, fit_robust
vol_stats = fit_robust(train, "close_vol")
move_stats = fit_robust(train, "odds_move_H")
ui_xgb = compute_ui(proba_xgb, test, vol_stats, move_stats)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1, 1]})
a1.hist(ui_xgb, bins=40, color="#4a6fa5", alpha=0.8)
a1.axvline(0.30, color="green", ls="--", lw=1, label="t_low=0.30")
a1.axvline(0.45, color="red", ls="--", lw=1, label="t_high=0.45")
a1.set_xlabel("Uncertainty Index (XGBoost)")
a1.set_ylabel("Matches")
a1.legend(fontsize=7, frameon=False)
a1.set_title("UI distribution (test)", fontsize=9)

ra = json.load(open(os.path.join(RES, "risk_analysis.json"), encoding="utf-8"))
tiers = ra["ui_tiers"]
labels = [k for k in ["low", "medium", "high(no-bet)"] if tiers[k].get("n", 0)]
accs = [tiers[k]["accuracy"] for k in labels]
a2.bar(labels, accs, color=["#4a9a6f", "#d9a441", "#c05555"], alpha=0.85)
for i, v in enumerate(accs):
    a2.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
a2.set_ylabel("Accuracy")
a2.set_ylim(0, 0.8)
a2.set_title("Accuracy by UI tier (XGBoost)", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_ui.png"))
plt.close()

# ============ 图4：累计 ROI 曲线 ============
print("[5/5] 累计 ROI 曲线 ...")
import glob
raw = pd.concat([pd.read_csv(p) for p in glob.glob(r"E:\论文\structured_data\*.csv")], ignore_index=True)
raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
tmeta = test.merge(raw[["Date", "HomeTeam", "AwayTeam", "B365CH", "B365CD", "B365CA"]],
                   on=["Date", "HomeTeam", "AwayTeam"], how="left")
odds = tmeta[["B365CH", "B365CD", "B365CA"]].values
from risk import risk_tiers

def cum_roi(proba, ui=None, t_low=0.30, t_hi=0.45, stop=False):
    rets = np.zeros(len(yte))
    pred = proba.argmax(axis=1)
    if ui is not None:
        tier, scale = risk_tiers(ui, t_low, t_hi)
    consec = 0
    peak = 1.0
    bank = 1.0
    dd_stop = False
    for i in range(len(yte)):
        if ui is not None and tier[i] == 2:
            consec = 0
            continue
        o = odds[i, pred[i]]
        if not np.isfinite(o) or o <= 1:
            continue
        s = 1.0
        if ui is not None:
            s = scale[i]
        if stop and dd_stop:
            s *= 0.2
        if pred[i] == yte[i]:
            r = s * (o - 1)
            consec = 0
        else:
            r = -s
            consec += 1
        bank += r
        peak = max(peak, bank)
        if stop and (consec >= 5 or (peak - bank) / peak >= 0.10):
            dd_stop = True
        rets[i] = r
    return np.cumsum(rets)

fig, ax = plt.subplots(figsize=(4.2, 2.8))
n = len(yte)
ax.plot(np.arange(n), cum_roi(proba_xgb), label="Naive (all bets)", lw=1)
ax.plot(np.arange(n), cum_roi(proba_xgb, ui_xgb), label="UI tiers + no-bet", lw=1)
ax.plot(np.arange(n), cum_roi(proba_xgb, ui_xgb, stop=True), label="+ stop-loss", lw=1)
ax.set_xlabel("Sequential bets (test season)")
ax.set_ylabel("Cumulative net return (units)")
ax.legend(fontsize=7, frameon=False)
ax.set_title("Cumulative returns by policy", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_roi.png"))
plt.close()

# ============ 图5：混淆矩阵 ============
print("[6/5] 混淆矩阵 ...")
da = json.load(open(os.path.join(RES, "draw_analysis.json"), encoding="utf-8"))
cm = np.array(da["xgb"]["confusion_matrix"])
fig, ax = plt.subplots(figsize=(3.4, 3.0))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(["H", "D", "A"])
ax.set_yticklabels(["H", "D", "A"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
ax.set_title("Confusion matrix (XGBoost, test)", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig5_confusion.png"))
plt.close()

# ============ 图6：coverage-accuracy / coverage-ROI 曲线（策略对比） ============
print("[6/6] coverage curves ...")
pc = json.load(open(os.path.join(RES, "policy_comparison.json"), encoding="utf-8"))
names = {"UI": "UI (uncertainty index)", "SCS": "SCS",
         "market_conf": "Market confidence", "model_conf": "Model confidence",
         "random": "Random filtering"}
colors = {"UI": "#1f77b4", "SCS": "#2ca02c", "market_conf": "#d62728",
          "model_conf": "#ff7f0e", "random": "#7f7f7f"}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
handles, labels = [], []
for name in names:
    curve = pc["strategies"][name]
    xs = [r["coverage"] for r in curve]
    accs = [r["acc"] * 100 for r in curve]
    rois = [r["roi"] * 100 for r in curve]
    l1, = axes[0].plot(xs, accs, label=names[name], lw=1.4, color=colors[name], marker="o", ms=2.5)
    axes[1].plot(xs, rois, label=names[name], lw=1.4, color=colors[name], marker="o", ms=2.5)
    handles.append(l1)
    labels.append(names[name])
axes[0].axhline(54.2, color="k", ls=":", lw=0.8)
axes[0].text(0.42, 55.0, "market acc.", fontsize=6.5, color="k")
axes[1].axhline(0, color="k", ls=":", lw=0.8)
for ax, ylab, title in [(axes[0], "Accuracy (%)", "Coverage vs accuracy"),
                        (axes[1], "ROI (%)", "Coverage vs ROI")]:
    ax.set_xlabel("Coverage (fraction of matches bet)")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0.38, 1.02)
    ax.grid(alpha=0.3)
# 共用底部图例，避免遮挡任意曲线
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7,
           frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.subplots_adjust(bottom=0.20)
plt.savefig(os.path.join(OUT, "fig6_coverage.png"))
plt.close()

print("\n图表已保存到", OUT)
