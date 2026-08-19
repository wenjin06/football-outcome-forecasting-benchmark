"""
Dixon-Coles 模型基线（向量化 MLE，快）
====================
- 分联赛拟合；λh = exp(μ + home + att_h + def_a), λa = exp(μ + att_a + def_h)
- τ 修正低比分（ρ）；L-BFGS-B 拟合；预测 test 与市场/XGB 同协议对比
输出：results/dixon_coles.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, log_loss

from evaluate import financial_metrics, simulate_bets

OUT = r"E:\论文\sci_redo\data\processed"
RES = r"E:\论文\sci_redo\results"
os.makedirs(RES, exist_ok=True)

raw = pd.concat([pd.read_csv(p) for p in glob.glob(r"E:\论文\structured_data\*.csv")], ignore_index=True)
raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
raw = raw.sort_values("Date").reset_index(drop=True)
MAX_GOALS = 8


def neg_ll_vec(params, team_codes, n_teams, home_idx, rho_idx, mu_idx, X, y, n_matches):
    att = np.concatenate([params[:n_teams - 1], [0.0]])
    deff = np.concatenate([params[n_teams - 1:2 * (n_teams - 1)], [0.0]])
    home = params[home_idx]
    rho = params[rho_idx]
    mu = params[mu_idx]
    hc = X[:, 0].astype(int)  # home team code
    ac = X[:, 1].astype(int)  # away team code
    lh = np.exp(mu + home + att[hc] + deff[ac])
    la = np.exp(mu + att[ac] + deff[hc])
    x, yg = y[:, 0], y[:, 1]
    ll = poisson.logpmf(x, lh) + poisson.logpmf(yg, la)
    # tau 修正（只影响 0/1 低比分组合）
    m00 = (x == 0) & (yg == 0)
    m01 = (x == 0) & (yg == 1)
    m10 = (x == 1) & (yg == 0)
    m11 = (x == 1) & (yg == 1)
    tau = np.ones(n_matches)
    tau[m00] = np.maximum(1 - lh[m00] * la[m00] * rho, 1e-8)
    tau[m01] = np.maximum(1 + lh[m01] * rho, 1e-8)
    tau[m10] = np.maximum(1 + la[m10] * rho, 1e-8)
    tau[m11] = np.maximum(1 - rho, 1e-8)
    ll = ll + np.log(tau)
    return -ll.sum()


def fit_dc(df):
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    n = len(teams)
    code = {t: i for i, t in enumerate(teams)}
    X = np.column_stack([df["HomeTeam"].map(code).values,
                         df["AwayTeam"].map(code).values])
    y = np.column_stack([df["FTHG"].values, df["FTAG"].values]).astype(float)
    home_idx = 2 * (n - 1)
    rho_idx = home_idx + 1
    mu_idx = rho_idx + 1
    x0 = np.zeros(mu_idx + 1)
    x0[home_idx] = 0.2
    x0[mu_idx] = np.log(np.mean(np.concatenate([df["FTHG"], df["FTAG"]])) + 1e-6)
    bounds = [(None, None)] * (mu_idx + 1)
    bounds[rho_idx] = (-0.15, 0.15)
    res = minimize(neg_ll_vec, x0, args=(code, n, home_idx, rho_idx, mu_idx, X, y, len(df)),
                   method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
    return teams, code, res.x, home_idx, rho_idx


def predict_dc(teams, code, params, home_idx, rho_idx, h_code, a_code):
    n = len(teams)
    att = np.concatenate([params[:n - 1], [0.0]])
    deff = np.concatenate([params[n - 1:2 * (n - 1)], [0.0]])
    home_adv = params[home_idx]
    rho = params[rho_idx]
    mu = params[-1]
    lh = np.exp(mu + home_adv + att[h_code] + deff[a_code])
    la = np.exp(mu + att[a_code] + deff[h_code])
    ph = np.array([poisson.pmf(i, lh) for i in range(MAX_GOALS)])
    pa = np.array([poisson.pmf(j, la) for j in range(MAX_GOALS)])
    m = np.outer(ph, pa)
    m[0, 0] *= max(1 - lh * la * rho, 1e-8)
    m[0, 1] *= max(1 + lh * rho, 1e-8)
    m[1, 0] *= max(1 + la * rho, 1e-8)
    m[1, 1] *= max(1 - rho, 1e-8)
    s = m.sum()
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()]) / s


train_raw = raw[raw["Date"] < "2024-08-01"]
test_raw = raw[raw["Date"] >= "2025-08-01"]

results = {}
proba_all = np.zeros((len(test_raw), 3))
unseen_count = 0
print("[1] 分联赛向量化拟合 Dixon-Coles ...", flush=True)
for div, sub in train_raw.groupby("Div"):
    t0 = pd.Timestamp.now()
    teams, code, params, hi, ri = fit_dc(sub)
    te = test_raw[test_raw["Div"] == div]
    ref = len(teams) - 1  # 参考队（att=def=0）
    probas = []
    for _, r in te.iterrows():
        h = code.get(r["HomeTeam"], ref)
        a = code.get(r["AwayTeam"], ref)
        if h == ref and r["HomeTeam"] not in code:
            unseen_count += 1
        if a == ref and r["AwayTeam"] not in code:
            unseen_count += 1
        probas.append(predict_dc(teams, code, params, hi, ri, h, a))
    proba_all[test_raw["Div"] == div] = np.array(probas)
    results[div] = {"teams": len(teams), "n_test": len(te)}
    print(f"  {div}: {len(teams)} 队 test {len(te)} 场 "
          f"({(pd.Timestamp.now()-t0).total_seconds():.0f}s)", flush=True)
print(f"未在训练集出现过的球队出场次数: {unseen_count}（按中性参数处理）", flush=True)

yte = test_raw["FTR"].map({"H": 0, "D": 1, "A": 2}).values
acc = accuracy_score(yte, proba_all.argmax(axis=1))
ll = log_loss(yte, proba_all, labels=[0, 1, 2])
from evaluate import brier_multiclass, ece
rets, placed = simulate_bets(yte, proba_all, test_raw["B365CH"].values,
                             test_raw["B365CD"].values, test_raw["B365CA"].values, min_prob=0.0)
fin = financial_metrics(rets, placed)
results["overall"] = {"acc": acc, "logloss": ll, "brier": float(brier_multiclass(yte, proba_all)),
                      "ece": float(ece(yte, proba_all)),
                      "roi": fin["roi"], "sharpe": fin["sharpe"], "mdd": fin["mdd"],
                      "n_bets": int(fin["n_bets"]), "win_rate": fin["win_rate"]}
print(f"\n整体: acc={acc:.4f} logloss={ll:.4f} ROI={fin['roi']*100:.2f}%")
print("（对照：Poisson 简化版 acc=0.508/logloss=1.003；市场 acc=0.542/logloss=0.967；XGB acc=0.542/logloss=0.971）")

with open(os.path.join(RES, "dixon_coles.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("已保存:", os.path.join(RES, "dixon_coles.json"))
