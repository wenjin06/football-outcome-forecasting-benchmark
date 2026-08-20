"""
LLM 预测实验（诚实版）
====================
用法：
  python src/run_llm.py --provider deepseek --n_matches 300 --n_samples 3
  python src/run_llm.py --provider local   --n_matches 200 --n_samples 2

- 从 test 集（2025/26 上半季）按时间序取前 n_matches 场（可选随机种子采样）
- 每场 n_samples 次推理；解析失败的样本计入 success_rate（如实报告）
- 多采样平均得到最终概率；consistency 用成对 TV 距离（risk.py）
- 评估：预测指标 + 财务模拟（同一投注协议）+ 成本估算
- 输出 results/llm_<provider>_t<temperature>.json + llm_<provider>_t<temperature>_per_match.csv
  （温度/供应商进文件名，避免不同配置互相覆盖）

说明：本地 qwen（WSL llama.cpp 8001）免费但慢，建议先用小样本验证流程；
DeepSeek 全量 1104 场 × 3 采样成本约 $1 量级，可接受。
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import evaluate_predictions, financial_metrics, simulate_bets
from risk import consistency_from_samples, compute_ui, fit_robust, compute_scs, risk_tiers, simulate_bets_with_risk

OUT = r"E:\论文\sci_redo\data\processed"
RES = r"E:\论文\sci_redo\results"
os.makedirs(RES, exist_ok=True)

DROP_COLS = ["Div", "Date", "Season", "HomeTeam", "AwayTeam", "FTR", "y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="deepseek", choices=["deepseek", "local"])
    ap.add_argument("--model", default=None,
                    help="Override the model name from llm-config.local.json "
                         "(in-memory only; the config file is never modified). "
                         "Example: --model deepseek-reasoner, or a local GGUF id.")
    ap.add_argument("--feature_mode", default="full",
                    choices=["full", "market", "stats", "market_stats"],
                    help="Which feature groups to include in the prompt "
                         "(LLM input ablation).")
    ap.add_argument("--n_matches", type=int, default=300)
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_mode", default="chronological",
                    choices=["chronological", "random"])
    args = ap.parse_args()

    from llm.llm_client import LLMClient
    from llm.prompts import build_feature_card
    feat = pd.read_csv(os.path.join(OUT, "all_matches_featurized.csv"),
                       parse_dates=["Date"])
    test = feat[feat["Date"] >= "2025-08-01"].sort_values("Date").reset_index(drop=True)

    if args.sample_mode == "random":
        test = test.sample(n=min(args.n_matches, len(test)), random_state=args.seed)
    else:
        test = test.head(args.n_matches)

    # 测试集赔率（财务模拟）
    import glob
    raw = pd.concat([pd.read_csv(p) for p in glob.glob(r"E:\论文\structured_data\*.csv")],
                    ignore_index=True)
    raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
    raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    tmeta = test.merge(raw[["Date", "HomeTeam", "AwayTeam", "B365CH", "B365CD", "B365CA"]],
                       on=["Date", "HomeTeam", "AwayTeam"], how="left")

    # train 用于归一化统计量（与管道一致）
    train = feat[feat["Date"] < "2024-08-01"]
    vol_stats = fit_robust(train, "close_vol")
    move_stats = fit_robust(train, "odds_move_H")

    client = LLMClient(provider=args.provider)
    if args.model:
        client.config[args.provider]["model"] = args.model
    model_tag = ""
    if args.model:
        model_tag = "_" + re.sub(r"[^A-Za-z0-9]", "-", args.model)
    fm_tag = "" if args.feature_mode == "full" else f"_fm{args.feature_mode}"
    print(f"provider={args.provider} matches={len(test)} samples={args.n_samples}")

    rows = []
    t0 = time.time()
    feat_cols_all = [c for c in test.columns if c not in DROP_COLS]
    ckpt_path = os.path.join(RES, f"llm_{args.provider}{model_tag}{fm_tag}_t{args.temperature}_s{args.n_samples}_partial.csv")
    # 断点续跑：若已有部分结果，加载并跳过已完成场次（只加载 idx 在当前范围内的）
    done_idx = set()
    if os.path.exists(ckpt_path):
        try:
            old = pd.read_csv(ckpt_path)
            for _, rr in old.iterrows():
                idx = int(rr["idx"])
                if idx >= len(test):
                    continue
                done_idx.add(idx)
                p = rr["p"]
                proba_old = np.array(json.loads(p)) if isinstance(p, str) and p else None
                rows.append({"idx": idx, "ok": int(rr["ok"]), "proba": proba_old,
                             "consistency": rr["consistency"] if "consistency" in rr else None})
            print(f"  检测到断点，已恢复 {len(rows)} 场（跳过已完成场次）")
        except Exception as e:
            print(f"  断点加载失败（{e}），从头开始")
    ...
    for i, (_, r) in enumerate(test.iterrows()):
        if i in done_idx:
            continue
        card = r[feat_cols_all].to_dict()  # 只含赛前特征，不含 y/结果/日期
        probs_list, ok, reasons = client.predict_match(card, n_samples=args.n_samples,
                                                       temperature=args.temperature,
                                                       feature_mode=args.feature_mode)
        n_ok = len([p for p in probs_list if p is not None])
        if n_ok == 0:
            rows.append({"idx": i, "ok": 0, "proba": None, "consistency": None})
            if i % 25 == 0:
                print(f"  [{i}/{len(test)}] FAILED (0/{args.n_samples} parsed)")
            continue
        valid = np.array([p for p in probs_list if p is not None])
        proba = valid.mean(axis=0)
        consistency = consistency_from_samples([p for p in probs_list if p is not None])[0] \
            if n_ok > 1 else 1.0
        rows.append({"idx": i, "ok": n_ok, "proba": proba, "consistency": consistency,
                     "reasons": reasons[:1]})
        if i % 25 == 0:
            print(f"  [{i}/{len(test)}] ok={n_ok}/{args.n_samples} "
                  f"p={proba.round(3)} cons={consistency:.2f}")
        # 断点保存：每 50 场写一次部分结果
        if i % 50 == 49:
            tmp = pd.DataFrame([{"idx": rw["idx"], "ok": rw["ok"],
                                 "p": json.dumps(rw["proba"].tolist()) if rw["proba"] is not None else None,
                                 "consistency": rw["consistency"]}
                                for rw in rows])
            tmp.to_csv(ckpt_path, index=False)

    done = [r for r in rows if r["ok"] > 0]
    failed = len(rows) - len(done)
    print(f"\n完成 {len(done)}/{len(rows)} 场，解析失败 {failed} 场 "
          f"(success_rate={len(done)/len(rows):.3f})")

    if not done:
        print("无有效预测，退出。")
        return

    y = test.loc[[r["idx"] for r in done], "y"].values.astype(int)
    proba = np.array([r["proba"] for r in done])
    consistency = np.array([r["consistency"] for r in done])
    tmeta_d = tmeta.iloc[[r["idx"] for r in done]]

    res = evaluate_predictions(y, proba)
    rets, placed = simulate_bets(y, proba, tmeta_d["B365CH"].values,
                                 tmeta_d["B365CD"].values, tmeta_d["B365CA"].values,
                                 min_prob=0.0)
    fin = financial_metrics(rets, placed)

    # 风控版本：UI 分层 + no-bet
    feat_d = test.iloc[[r["idx"] for r in done]].reset_index(drop=True)
    ui = compute_ui(proba, feat_d, vol_stats, move_stats, consistency=consistency)
    scs = compute_scs(feat_d, vol_stats, move_stats)
    from risk import simulate_bets_with_risk
    rets_r, placed_r, scales_r = simulate_bets_with_risk(
        y, proba, tmeta_d["B365CH"].values, tmeta_d["B365CD"].values,
        tmeta_d["B365CA"].values, ui, t_low=0.4, t_hi=0.7)
    fin_r = financial_metrics(rets_r, placed_r)
    tier_r, _ = risk_tiers(ui, 0.4, 0.7)
    n_bet = int(placed_r.sum())
    n_nobet = int((tier_r == 2).sum())

    summary = {
        "provider": args.provider, "n_matches_total": len(rows),
        "temperature": args.temperature, "n_samples": args.n_samples,
        "n_ok": len(done), "success_rate": len(done) / len(rows),
        "accuracy": res["accuracy"], "macro_f1": res["macro_f1"],
        "log_loss": res["log_loss"], "brier": res["brier"], "ece": res["ece"],
        "fin_all": fin,
        "fin_risk": {"roi": fin_r["roi"], "sharpe": fin_r["sharpe"],
                     "mdd": fin_r["mdd"], "win_rate": fin_r["win_rate"],
                     "n_bets": n_bet, "n_no_bet": n_nobet,
                     "coverage": n_bet / len(y)},
        "consistency_mean": float(consistency.mean()),
        "cost_usd": client.estimate_cost(),
        "tokens": {"prompt": client.total_prompt_tokens,
                   "completion": client.total_completion_tokens},
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(RES, f"llm_{args.provider}{model_tag}{fm_tag}_t{args.temperature}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    rec = pd.DataFrame({
        "idx": [r["idx"] for r in done],
        "y": y, "ok": [r["ok"] for r in done],
        "p_H": proba[:, 0], "p_D": proba[:, 1], "p_A": proba[:, 2],
        "consistency": consistency, "ui": ui, "scs": scs,
        "odds_H": tmeta_d["B365CH"].values, "odds_D": tmeta_d["B365CD"].values,
        "odds_A": tmeta_d["B365CA"].values,
    })
    rec.to_csv(os.path.join(RES, f"llm_{args.provider}{model_tag}{fm_tag}_t{args.temperature}_per_match.csv"), index=False)

    print(f"\n===== LLM ({args.provider}{model_tag}{fm_tag}) =====")
    print(f"  acc={res['accuracy']:.4f} macroF1={res['macro_f1']:.4f} "
          f"logloss={res['log_loss']:.4f} brier={res['brier']:.4f} ece={res['ece']:.4f}")
    print(f"  全下注: ROI={fin['roi']*100:.2f}% Sharpe={fin['sharpe']:.3f} "
          f"MDD={fin['mdd']*100:.1f}% 胜率={fin['win_rate']*100:.1f}% ({fin['n_bets']}注)")
    print(f"  风控后: ROI={fin_r['roi']*100:.2f}% Sharpe={fin_r['sharpe']:.3f} "
          f"MDD={fin_r['mdd']*100:.1f}% 胜率={fin_r['win_rate']*100:.1f}% "
          f"({n_bet}注, no-bet {n_nobet}, 覆盖率={n_bet/len(y):.2f})")
    print(f"  cost=${summary['cost_usd']:.2f} elapsed={summary['elapsed_s']}s")
    print("已保存:", os.path.join(RES, f"llm_{args.provider}{model_tag}{fm_tag}_t{args.temperature}.json"))


if __name__ == "__main__":
    main()
