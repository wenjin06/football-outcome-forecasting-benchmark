"""
Risk-control module: revised SCS + revised UI + tiered position sizing + no-bet
==============================================================================
Corresponds to required revisions #3 (SCS direction error) and #4 (UI lacked a
consistency term and had unjustified weights).

Design principles (final version):
1. All components are normalized to [0,1] with a consistent direction:
   larger = more uncertain/more complex
2. Normalization statistics are fitted on train only; val/test are
   transformed only (consistent with the data pipeline, no leakage)
3. Thresholds and weights are determined on val (grid search); test is used
   only for final evaluation, with threshold sensitivity reported
4. No-bet samples are excluded from financial metrics, but their prediction
   accuracy is reported separately

SCS (revised):
    SCS = (1 - gap_norm) + vol_norm + move_norm
  - gap_norm: home-away strength gap, normalized by the pre-match league rank
    difference (|H_rank - A_rank| / number of teams in the league)
    Letting SCS increase with |S_home - S_away| was a direction error: a larger
    strength gap makes a match easier to predict, so the term is (1 - gap_norm).
  - vol_norm: dispersion of closing odds across bookmakers (close_vol),
    robust-normalized on train
  - move_norm: magnitude of closing-to-opening odds movement (market reaction
    to new information), robust-normalized

UI (revised, all terms in [0,1], includes multi-sample consistency):
    UI = w_pred*(1 - p_max) + w_cons*(1 - consistency) + w_vol*vol_norm + w_move*move_norm
  - p_max: maximum model probability (inverse confidence term)
  - consistency: multi-sample agreement. For the LLM, the inverse of the mean
    pairwise TV distance of probabilities across repeated inferences;
    deterministic models (XGB/RF/rules) have consistency = 1 (no sampling disagreement)
  - vol_norm / move_norm: market-volatility terms (as above)
  - Default weights (0.4, 0.3, 0.15, 0.15), overridable by grid search on val

Risk-control policy:
    UI <= t_low        : full stake (position scale 1.0)
    t_low < UI <= t_hi : linear stake reduction (1.0 -> 0.2)
    UI > t_hi          : no-bet (excluded from ROI; accuracy reported separately)
"""
import numpy as np
import pandas as pd


def rank_gap_norm(meta_or_feat):
    """Home-away rank-gap normalization: |H_rank - A_rank| / (teams per league - 1). No fit; deterministic."""
    TEAMS_PER_LEAGUE = 20  # all top-five leagues have 20 teams
    hr = meta_or_feat.get("H_rank", None)
    ar = meta_or_feat.get("A_rank", None)
    if hr is None or ar is None:
        return np.full(len(meta_or_feat), np.nan)
    gap = (hr - ar).abs()
    return (gap / (TEAMS_PER_LEAGUE - 1)).clip(0, 1)


def robust_norm(values, med, iqr):
    """Robust normalization: (x - med) / iqr, clipped to [0,1]. med/iqr come from train only."""
    x = (values - med) / iqr
    return x.clip(0, 1)


def fit_robust(train_feat, col):
    """Compute median/IQR from train (fitted once)."""
    s = train_feat[col].dropna()
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    if iqr <= 0:
        iqr = s.std() or 1.0
    return med, iqr


def compute_scs(feat, vol_stats, move_stats):
    """
    feat: DataFrame containing H_rank/A_rank/close_vol/odds_move_H/odds_move_A
    vol_stats/move_stats: (med, iqr) from train
    Returns an SCS array in [0,3]; higher = more complex.
    """
    gap = rank_gap_norm(feat)
    vol = robust_norm(feat["close_vol"], *vol_stats)
    move = robust_norm(
        (feat["odds_move_H"].abs() + feat["odds_move_A"].abs()), *move_stats)
    scs = (1 - gap.fillna(0.5)) + vol.fillna(0.5) + move.fillna(0.5)
    return scs.values


def consistency_from_samples(proba_samples):
    """
    proba_samples: list of probability vectors (n,3) from repeated inference on
    the same input. Returns a consistency array in [0,1]: 1 = fully consistent.
    Mean pairwise TV distance: TV = 0.5 * sum|p-q|; consistency = 1 - mean(TV).
    Deterministic models pass proba_samples=[proba] (single sample -> consistency=1).
    """
    if len(proba_samples) == 1:
        return np.ones(len(proba_samples[0]))
    arr = np.stack(proba_samples, axis=0)  # (S, n, 3); for single-match calls n=1 -> (S, 3)
    if arr.ndim == 2:
        arr = arr[:, None, :]  # Unify to (S, 1, 3)
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
    proba: (n,3) probability matrix
    feat: DataFrame containing close_vol/odds_move_H/odds_move_A
    consistency: array in [0,1]; None is treated as all-ones (deterministic models)
    w: (w_pred, w_cons, w_vol, w_move); must sum to 1
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
    Returns (tier, position_scale, no_bet)
    tier: 0=low, 1=medium, 2=high(no-bet)
    position_scale: position-size factor in [0,1]
    """
    tier = np.where(ui <= t_low, 0, np.where(ui <= t_hi, 1, 2))
    scale = np.where(tier == 0, 1.0, np.where(tier == 1,
                    (t_hi - ui) / (t_hi - t_low) * 0.8 + 0.2, 0.0))
    return tier, scale


def simulate_bets_with_risk(y_true, proba, odds_h, odds_d, odds_a,
                            ui, t_low=0.4, t_hi=0.7, stake=1.0,
                            stop_loss_n=5, stop_loss_dd=0.10):
    """
    Betting simulation with risk control:
    - no-bet (UI > t_hi): no stake, excluded from ROI
    - position size = position_scale
    - stop-loss: after stop_loss_n consecutive losses or a cumulative drawdown
      exceeding stop_loss_dd, all subsequent stakes are reduced to 0.2x
    Returns (returns, placed, position_scales)
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
