"""
基线模型：市场基线(de-vig) / Poisson / Elo / XGBoost / RandomForest
全部在无泄漏管道产物上训练与评估。
"""
import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from evaluate import (evaluate_predictions, financial_metrics, simulate_bets,
                      class_metrics, bootstrap_ci)

OUT = r"E:\论文\sci_redo\data\processed"
RES = r"E:\论文\sci_redo\results"
os.makedirs(RES, exist_ok=True)

train = joblib.load(os.path.join(OUT, "train_dataset.pkl"))
val = joblib.load(os.path.join(OUT, "val_dataset.pkl"))
test = joblib.load(os.path.join(OUT, "test_dataset.pkl"))

# 原始数据（用于赔率）
raw = pd.concat([pd.read_csv(p) for p in glob.glob(r"E:\论文\structured_data\*.csv")], ignore_index=True)
raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
feat = pd.read_csv(r"E:\论文\sci_redo\data\processed\all_matches_featurized.csv", parse_dates=["Date"])
# 按 Date+HomeTeam+AwayTeam 对齐测试集赔率（从原始数据取收盘赔率）
tmeta = test["meta"].copy()
tmeta = tmeta.merge(raw[["Date", "HomeTeam", "AwayTeam", "B365CH", "B365CD", "B365CA"]],
                    on=["Date", "HomeTeam", "AwayTeam"], how="left")
tmeta = tmeta.merge(feat[["Date", "HomeTeam", "AwayTeam",
                          "mkt_prob_H", "mkt_prob_D", "mkt_prob_A"]],
                    on=["Date", "HomeTeam", "AwayTeam"], how="left")

def report(name, proba, meta_df, odds_df):
    y = meta_df["y"].values if (meta_df is not None and "y" in meta_df.columns) else test["target"]
    res = evaluate_predictions(y, proba)
    # 财务模拟：收盘赔率
    rets, placed = simulate_bets(y, proba, odds_df["B365CH"].values, odds_df["B365CD"].values,
                                 odds_df["B365CA"].values, min_prob=0.0)
    fin = financial_metrics(rets, placed)
    # 仅当概率>0.4 才下注（风险控制最小版本）
    rets2, placed2 = simulate_bets(y, proba, odds_df["B365CH"].values, odds_df["B365CD"].values,
                                   odds_df["B365CA"].values, min_prob=0.4)
    fin2 = financial_metrics(rets2, placed2)

    # bootstrap CI（accuracy + ROI）
    n = len(y)
    rng = np.random.default_rng(42)
    acc_boot, roi_boot = [], []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        acc_boot.append(accuracy_score(y[idx], proba[idx].argmax(axis=1)))
        if placed2.sum() > 0:
            r_sub = rets2[idx]
            p_sub = placed2[idx]
            if p_sub.sum() > 0:
                roi_boot.append(r_sub[p_sub].sum() / p_sub.sum())
    acc_ci = (np.percentile(acc_boot, 2.5), np.percentile(acc_boot, 97.5))
    roi_ci = (np.percentile(roi_boot, 2.5), np.percentile(roi_boot, 97.5)) if roi_boot else (np.nan, np.nan)

    print(f"\n===== {name} =====")
    print(f"  Accuracy={res['accuracy']:.4f} [{acc_ci[0]:.4f},{acc_ci[1]:.4f}] MacroF1={res['macro_f1']:.4f} LogLoss={res['log_loss']:.4f} Brier={res['brier']:.4f} ECE={res['ece']:.4f}")
    print(f"  全下注: ROI={fin['roi']*100:.2f}% Sharpe={fin['sharpe']:.3f} MDD={fin['mdd']*100:.1f}% 胜率={fin['win_rate']*100:.1f}% ({fin['n_bets']}注)")
    print(f"  阈值0.4: ROI={fin2['roi']*100:.2f}% [{roi_ci[0]*100:.2f},{roi_ci[1]*100:.2f}] Sharpe={fin2['sharpe']:.3f} MDD={fin2['mdd']*100:.1f}% 胜率={fin2['win_rate']*100:.1f}% ({fin2['n_bets']}注)")
    return {"name": name, **res, "acc_ci": acc_ci, "roi_ci": roi_ci, "fin_all": fin, "fin_t04": fin2}

results = {}

# ============ 1. 市场基线：收盘赔率 de-vig ============
print("[1] 市场基线 (收盘赔率 de-vig)")
mkt_proba = tmeta[["mkt_prob_H", "mkt_prob_D", "mkt_prob_A"]].values
results["market"] = report("Market (de-vig)", mkt_proba, None, tmeta)

# ============ 2. XGBoost ============
print("\n[2] XGBoost")
xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                    early_stopping_rounds=30, random_state=42)
xgb.fit(train["features"], train["target"], eval_set=[(val["features"], val["target"])], verbose=False)
proba_xgb = xgb.predict_proba(test["features"])
results["xgb"] = report("XGBoost", proba_xgb, None, tmeta)
joblib.dump(xgb, os.path.join(RES, "xgb_model.pkl"))

# ============ 3. Random Forest ============
print("\n[3] Random Forest")
rf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(train["features"], train["target"])
proba_rf = rf.predict_proba(test["features"])
results["rf"] = report("RandomForest", proba_rf, None, tmeta)
joblib.dump(rf, os.path.join(RES, "rf_model.pkl"))

# ============ 4. Elo（仅用历史结果+主场优势） ============
print("\n[4] Elo")
def elo_features(all_matches):
    """Elo 评分：只用赛前可得信息（遍历历史，逐场更新）"""
    elo = {}
    K = 32
    home_adv = 60
    rows = []
    for _, r in all_matches.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        eh = elo.get(h, 1500) + home_adv
        ea = elo.get(a, 1500)
        we_h = 1 / (1 + 10 ** ((ea - eh) / 400))
        we_a = 1 - we_h
        # 赛前特征：双方 Elo 及差值
        rows.append({"Date": r["Date"], "HomeTeam": h, "AwayTeam": a,
                     "elo_h": eh, "elo_a": ea, "elo_diff": eh - ea,
                     "FTR": r["FTR"]})
        ftr = r["FTR"]
        s_h = 1 if ftr == "H" else (0.5 if ftr == "D" else 0)
        s_a = 1 - s_h
        elo[h] = elo.get(h, 1500) + K * (s_h - we_h)
        elo[a] = elo.get(a, 1500) + K * (s_a - we_a)
    return pd.DataFrame(rows)

elo_df = elo_features(raw)
# 对齐测试集
test_meta = test["meta"].copy()
test_meta = test_meta.merge(elo_df, on=["Date", "HomeTeam", "AwayTeam"], how="left")
# Elo 概率（多项逻辑回归校准）
from sklearn.linear_model import LogisticRegression
elo_df2 = elo_df.rename(columns={"FTR": "elo_FTR"})
train_meta = train["meta"].rename(columns={"FTR": "meta_FTR"})
elo_train = elo_df2.merge(train_meta[["Date", "HomeTeam", "AwayTeam", "meta_FTR"]],
                          on=["Date", "HomeTeam", "AwayTeam"], how="inner").dropna(subset=["elo_diff"])
# 用训练集拟合 Elo 差 -> 三分类概率（多项逻辑回归）
lr = LogisticRegression(max_iter=1000)
lr.fit(elo_train[["elo_diff"]].values, elo_train["meta_FTR"].map({"H": 0, "D": 1, "A": 2}).values)
test_meta = test["meta"].copy()
test_meta = test_meta.merge(elo_df2, on=["Date", "HomeTeam", "AwayTeam"], how="left")
diff = test_meta["elo_diff"].fillna(0).values
proba_elo = lr.predict_proba(diff.reshape(-1, 1))
results["elo"] = report("Elo", proba_elo, None, tmeta)

# ============ 5. Poisson（简化：用 H/A 场均进球） ============
print("\n[5] Poisson")
# 训练集每队场均进球/失球
train_raw = raw[raw["Date"] < "2024-08-01"]
home_avg = train_raw.groupby("HomeTeam")["FTHG"].mean()
away_avg = train_raw.groupby("AwayTeam")["FTAG"].mean()
league_home = train_raw["FTHG"].mean()
league_away = train_raw["FTAG"].mean()

def poisson_proba(lam_h, lam_a):
    from scipy.stats import poisson
    max_goals = 8
    ph = np.array([poisson.pmf(i, lam_h) for i in range(max_goals)])
    pa = np.array([poisson.pmf(i, lam_a) for i in range(max_goals)])
    m = np.outer(ph, pa)
    p_home = np.tril(m, -1).sum()
    p_draw = np.trace(m)
    p_away = np.triu(m, 1).sum()
    return np.array([p_home, p_draw, p_away])

proba_poisson = []
for _, r in test["meta"].iterrows():
    lh = home_avg.get(r["HomeTeam"], league_home) * 1.1  # 主场增益
    la = away_avg.get(r["AwayTeam"], league_away) * 1.1
    proba_poisson.append(poisson_proba(lh, la))
proba_poisson = np.array(proba_poisson)
results["poisson"] = report("Poisson", proba_poisson, None, tmeta)

# ============ 保存汇总 ============
summary = {}
for k, v in results.items():
    summary[k] = {
        "accuracy": v["accuracy"], "acc_ci": v["acc_ci"], "macro_f1": v["macro_f1"],
        "log_loss": v["log_loss"], "brier": v["brier"], "ece": v["ece"],
        "roi_all": v["fin_all"]["roi"], "sharpe_all": v["fin_all"]["sharpe"],
        "mdd_all": v["fin_all"]["mdd"], "winrate_all": v["fin_all"]["win_rate"],
        "n_bets_all": v["fin_all"]["n_bets"],
        "roi_t04": v["fin_t04"]["roi"], "roi_ci_t04": v["roi_ci"],
        "sharpe_t04": v["fin_t04"]["sharpe"],
        "mdd_t04": v["fin_t04"]["mdd"], "winrate_t04": v["fin_t04"]["win_rate"],
        "n_bets_t04": v["fin_t04"]["n_bets"],
    }
with open(os.path.join(RES, "baselines_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
print("\n\n===== 汇总表 =====")
df_sum = pd.DataFrame(summary).T
print(df_sum.round(3).to_string())
print("\n已保存:", os.path.join(RES, "baselines_summary.json"))
