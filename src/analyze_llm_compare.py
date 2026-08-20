"""
LLM 对比分析：温度敏感性 + 开源/闭源公平对比
====================
1. 温度敏感性：DeepSeek t=0.7 vs t=0.3 在同一批 120 场上
   - acc/logloss/ECE/一致性分布/一致性-正确性相关
2. 开源对照：本地 qwen vs DeepSeek 在同一批 200 场上
   - acc/logloss/ECE/ROI（等注协议）
输出：results/llm_compare.json
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from evaluate import brier_multiclass, ece, financial_metrics, simulate_bets

RES = r"E:\论文\sci_redo\results"


def load(name):
    df = pd.read_csv(os.path.join(RES, name))
    df = df.dropna(subset=["p_H"]).reset_index(drop=True)
    proba = df[["p_H", "p_D", "p_A"]].values
    return df, proba


def metrics(y, proba, odds=None):
    out = {
        "acc": float(accuracy_score(y, proba.argmax(axis=1))),
        "logloss": float(log_loss(y, proba, labels=[0, 1, 2])),
        "brier": float(brier_multiclass(y, proba)),
        "ece": float(ece(y, proba)),
        "n": int(len(y)),
    }
    if odds is not None:
        rets, placed = simulate_bets(y, proba, odds[:, 0], odds[:, 1], odds[:, 2], min_prob=0.0)
        fin = financial_metrics(rets, placed)
        out["roi"] = fin["roi"]
        out["win_rate"] = fin["win_rate"]
    return out


def consistency_corr(df):
    """一致性-正确性分析：分箱 + 点二列相关。"""
    correct = (df[["p_H", "p_D", "p_A"]].values.argmax(axis=1) == df["y"].values).astype(float)
    cons = df["consistency"].values
    bins = [0, 0.7, 0.8, 0.9, 0.95, 1.0]
    table = []
    for i in range(len(bins) - 1):
        mask = (cons >= bins[i]) & (cons < bins[i + 1])
        if mask.sum() < 10:
            continue
        table.append({"bin": f"[{bins[i]},{bins[i+1]})", "n": int(mask.sum()),
                      "acc": float(correct[mask].mean())})
    return {"bins": table, "point_biserial": float(np.corrcoef(cons, correct)[0, 1]),
            "cons_mean": float(cons.mean()), "cons_frac_ge095": float((cons >= 0.95).mean())}


out = {}

# ============ 1. 温度敏感性（同批 120 场） ============
d03, p03 = load("llm_deepseek_t0.3_per_match.csv")
d07, p07 = load("llm_deepseek_t0.7_per_match.csv")

sub03 = d03.head(120).reset_index(drop=True)
p03_120 = sub03[["p_H", "p_D", "p_A"]].values
y120 = sub03["y"].values.astype(int)
odds120 = sub03[["odds_H", "odds_D", "odds_A"]].values

out["temperature"] = {
    "t0.3": metrics(y120, p03_120, odds120),
    "t0.7": metrics(y120, p07, None),
    "t0.3_consistency": consistency_corr(sub03),
    "t0.7_consistency": consistency_corr(d07),
}
print("=== 温度敏感性（同批 120 场） ===")
for k in ["t0.3", "t0.7"]:
    m = out["temperature"][k]
    c = out["temperature"][f"{k}_consistency"]
    print(f"  {k}: acc={m['acc']:.3f} logloss={m['logloss']:.4f} ece={m['ece']:.4f} "
          f"cons_mean={c['cons_mean']:.3f} cons>=0.95占比={c['cons_frac_ge095']:.2f} "
          f"cons-correct r={c['point_biserial']:.3f}")

# ============ 2. 开源/模型对照（同批场次） ============
dl, pl = load("llm_local_per_match.csv")
sub03_200 = d03.head(200).reset_index(drop=True)
p03_200 = sub03_200[["p_H", "p_D", "p_A"]].values
y200 = sub03_200["y"].values.astype(int)
odds200 = sub03_200[["odds_H", "odds_D", "odds_A"]].values
yl = dl["y"].values.astype(int)
oddsl = dl[["odds_H", "odds_D", "odds_A"]].values

out["open_vs_closed"] = {
    "deepseek_t0.3": metrics(y200, p03_200, odds200),
    "qwen_local": metrics(yl, pl, oddsl),
}
print("\n=== 模型对照（deepseek-chat 200 场 / qwen 200 场） ===")
for k in ["deepseek_t0.3", "qwen_local"]:
    m = out["open_vs_closed"][k]
    print(f"  {k}: acc={m['acc']:.3f} logloss={m['logloss']:.4f} ece={m['ece']:.4f} "
          f"roi={m['roi']*100:.2f}% win={m['win_rate']:.3f}")

# reasoner（推理模型）对照：同批前 120 场（n_samples=1，耗时限制）
import os as _os
reasoner_path = _os.path.join(RES, "llm_deepseek_deepseek-reasoner_t0.3_per_match.csv")
if _os.path.exists(reasoner_path):
    dr, pr = load("llm_deepseek_deepseek-reasoner_t0.3_per_match.csv")
    sub03_120 = d03.head(120).reset_index(drop=True)
    p03_120b = sub03_120[["p_H", "p_D", "p_A"]].values
    y120b = sub03_120["y"].values.astype(int)
    odds120b = sub03_120[["odds_H", "odds_D", "odds_A"]].values
    subq_120 = dl.head(120).reset_index(drop=True)
    pq_120 = subq_120[["p_H", "p_D", "p_A"]].values
    yq_120 = subq_120["y"].values.astype(int)
    oddsq_120 = subq_120[["odds_H", "odds_D", "odds_A"]].values
    yr = dr["y"].values.astype(int)
    oddsr = dr[["odds_H", "odds_D", "odds_A"]].values
    out["open_vs_closed"]["deepseek_reasoner"] = metrics(yr, pr, oddsr)
    out["reasoner_notes"] = {
        "n_matches": int(len(dr)), "n_samples": 1,
        "comparison": "same first 120 matches of the test set "
                       "(chronological order), all three models",
        "deepseek_t0.3_same_120": metrics(y120b, p03_120b, odds120b),
        "qwen_local_same_120": metrics(yq_120, pq_120, oddsq_120),
    }
    print("\n=== 推理模型对照（同批前 120 场，三模型） ===")
    for k in ["deepseek_t0.3_same_120", "deepseek_reasoner", "qwen_local_same_120"]:
        m = out["reasoner_notes"][k] if k.endswith("_same_120") else out["open_vs_closed"][k]
        print(f"  {k}: acc={m['acc']:.3f} logloss={m['logloss']:.4f} ece={m['ece']:.4f} "
              f"roi={m['roi']*100:.2f}% win={m['win_rate']:.3f}")
else:
    print("\n[skip] reasoner 对照未运行（缺少 llm_deepseek_deepseek-reasoner_t0.3_per_match.csv）")

with open(os.path.join(RES, "llm_compare.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=float)
print("\n已保存:", os.path.join(RES, "llm_compare.json"))
