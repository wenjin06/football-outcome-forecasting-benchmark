"""
风控模块：修正版 SCS + 修正版 UI + 分层仓位 + no-bet
====================

设计原则（诚实版）：
1. 所有成分归一化到 [0,1]，方向统一为"越大 = 越不确定/越复杂"
2. 归一化统计量只在 train 上 fit，val/test 仅 transform（与数据管道一致，无泄漏）
3. 阈值与权重在 val 上确定（网格搜索），test 只做最终评估，报告阈值敏感性

SCS（修正版）：
    SCS = (1 - gap_norm) + vol_norm + move_norm
  - gap_norm: 主客实力差，用赛前联赛排名差归一化（|H_rank - A_rank| / 联赛球队数）
    原论文 |S_home - S_away| 越大 SCS 越大是方向错误：实力差越大结果越可预测，
    故修正为 (1 - gap_norm)。
  - vol_norm: 多家博彩收盘赔率离散度（close_vol），train 上 robust 归一化
  - move_norm: 收盘-初盘赔率变动幅度（市场对突发信息的反应），robust 归一化

UI（修正版，全部项 [0,1]，含多采样一致性）：
    UI = w_pred*(1 - p_max) + w_cons*(1 - consistency) + w_vol*vol_norm + w_move*move_norm
  - p_max: 模型输出最大概率（置信度反向项）
  - consistency: 多采样一致度。LLM 多次推理概率的平均成对 TV 距离反向；
    确定性模型（XGB/RF/规则）consistency = 1（无采样分歧）
  - vol_norm / move_norm: 市场波动项（同上）
  - 默认权重 (0.4, 0.3, 0.15, 0.15)，可在 val 上网格搜索后覆盖

风控策略：
    UI <= t_low        : 全仓（仓位系数 1.0）
    t_low < UI <= t_hi : 线性缩仓（1.0 -> 0.2）
    UI > t_hi          : no-bet（不计入 ROI，预测精度单独报告）
"""
import numpy as np
import pandas as pd


def rank_gap_norm(meta_or_feat):
    """主客排名差归一化：|H_rank - A_rank| / (联赛球队数 - 1)。无 fit，确定性。"""
    TEAMS_PER_LEAGUE = 20  # 五大联赛均为 20 队
    hr = meta_or_feat.get("H_rank", None)
    ar = meta_or_feat.get("A_rank", None)
    if hr is None or ar is None:
        return np.full(len(meta_or_feat), np.nan)
    gap = (hr - ar).abs()
    return (gap / (TEAMS_PER_LEAGUE - 1)).clip(0, 1)


def robust_norm(values, med, iqr):
    """robust 归一化：(x - med) / iqr，clip 到 [0,1]。med/iqr 只来自 train。"""
    x = (values - med) / iqr
    return x.clip(0, 1)


def fit_robust(train_feat, col):
    """从 train 计算 median/IQR（只 fit 一次）。"""
    s = train_feat[col].dropna()
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    if iqr <= 0:
        iqr = s.std() or 1.0
    return med, iqr


def compute_scs(feat, vol_stats, move_stats):
    """
    feat: 含 H_rank/A_rank/close_vol/odds_move_H/odds_move_A 的 DataFrame
    vol_stats/move_stats: (med, iqr) 来自 train
    返回 SCS 数组 [0,3]，越高越复杂。
    """
    gap = rank_gap_norm(feat)
    vol = robust_norm(feat["close_vol"], *vol_stats)
    move = robust_norm(
        (feat["odds_move_H"].abs() + feat["odds_move_A"].abs()), *move_stats)
    scs = (1 - gap.fillna(0.5)) + vol.fillna(0.5) + move.fillna(0.5)
    return scs.values


def consistency_from_samples(proba_samples):
    """
    proba_samples: list of 概率向量 (n,3)，来自同一输入的多采样推理。
    返回 consistency 数组 [0,1]：1 = 完全一致。
    平均成对 TV 距离：TV = 0.5 * sum|p-q|；consistency = 1 - mean(TV)。
    确定性模型直接传 proba_samples=[proba]（单样本 -> consistency=1）。
    """
    if len(proba_samples) == 1:
        return np.ones(len(proba_samples[0]))
    arr = np.stack(proba_samples, axis=0)  # (S, n, 3)；单场调用时 n=1 -> (S, 3)
    if arr.ndim == 2:
        arr = arr[:, None, :]  # 统一为 (S, 1, 3)
    S = arr.shape[0]
    tvs = []
    for i in range(S):
        for j in range(i + 1, S):
            tvs.append(0.5 * np.abs(arr[i] - arr[j]).sum(axis=1))
    mean_tv = np.mean(tvs, axis=0)
    return 1.0 - mean_tv


def compute_ui(proba, feat, vol_stats, move_stats, consistency=None,
               w=(0.4, 0.3, 0.15, 0.15)):
    """
    proba: (n,3) 概率矩阵
    feat: 含 close_vol/odds_move_H/odds_move_A 的 DataFrame
    consistency: 数组 [0,1]；None 时视为全 1（确定性模型）
    w: (w_pred, w_cons, w_vol, w_move)，和须为 1
    """
    p_max = proba.max(axis=1)
    if consistency is None:
        consistency = np.ones(len(proba))
    vol = robust_norm(feat["close_vol"], *vol_stats)
    move = robust_norm(
        (feat["odds_move_H"].abs() + feat["odds_move_A"].abs()), *move_stats)
    w_pred, w_cons, w_vol, w_move = w
    ui = (w_pred * (1 - p_max)
          + w_cons * (1 - consistency)
          + w_vol * vol.fillna(0.5).values
          + w_move * move.fillna(0.5).values)
    return ui


def risk_tiers(ui, t_low=0.4, t_hi=0.7):
    """
    返回 (tier, position_scale, no_bet)
    tier: 0=low, 1=medium, 2=high(no-bet)
    position_scale: 仓位系数 [0,1]
    """
    tier = np.where(ui <= t_low, 0, np.where(ui <= t_hi, 1, 2))
    scale = np.where(tier == 0, 1.0, np.where(tier == 1,
                    (t_hi - ui) / (t_hi - t_low) * 0.8 + 0.2, 0.0))
    return tier, scale


def simulate_bets_with_risk(y_true, proba, odds_h, odds_d, odds_a,
                            ui, t_low=0.4, t_hi=0.7, stake=1.0,
                            stop_loss_n=5, stop_loss_dd=0.10):
    """
    带风控的投注模拟：
    - no-bet（UI > t_hi）不下注，不计入 ROI
    - 仓位系数 = position_scale
    - 止损：连续亏损 stop_loss_n 笔，或累计回撤超 stop_loss_dd 时，后续全部降为 0.2 倍仓位
    返回 (returns, placed, position_scales)
    """
    n = len(y_true)
    returns = np.zeros(n)
    placed = np.zeros(n, dtype=bool)
    scales = np.zeros(n)
    tier, scale = risk_tiers(ui, t_low, t_hi)

    pred = proba.argmax(axis=1)
    consec_loss = 0
    bankroll = 1.0
    peak = 1.0
    dd_stop = False

    for i in range(n):
        if tier[i] == 2:  # no-bet
            consec_loss = 0
            continue
        odds = [odds_h[i], odds_d[i], odds_a[i]][pred[i]]
        if not np.isfinite(odds) or odds <= 1:
            continue
        s = scale[i] * stake
        if dd_stop:
            s *= 0.2
        placed[i] = True
        scales[i] = s
        if pred[i] == y_true[i]:
            ret = s * (odds - 1)
            consec_loss = 0
        else:
            ret = -s
            consec_loss += 1
        returns[i] = ret
        bankroll += ret
        peak = max(peak, bankroll)
        if consec_loss >= stop_loss_n or (peak - bankroll) / peak >= stop_loss_dd:
            dd_stop = True
    return returns, placed, scales
