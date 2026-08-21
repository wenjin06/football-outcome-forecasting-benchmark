"""
Evaluation utilities: metrics + bootstrap confidence intervals
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

def brier_multiclass(y_true, proba):
    """Multiclass Brier score (scalar version)"""
    K = proba.shape[1]
    y_onehot = np.zeros((len(y_true), K))
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    return np.mean(np.sum((proba - y_onehot) ** 2, axis=1))

def ece(y_true, proba, n_bins=10):
    """Expected Calibration Error (binned by maximum predicted probability)"""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_true)) * abs(acc[mask].mean() - conf[mask].mean())
    return ece

def reliability_table(y_true, proba, n_bins=10):
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        rows.append({"bin": f"({bins[i]:.1f},{bins[i+1]:.1f}]", "n": int(mask.sum()),
                     "acc": acc[mask].mean(), "conf": conf[mask].mean()})
    return pd.DataFrame(rows)

def bootstrap_ci(values, metric_fn, n_boot=2000, seed=42, alpha=0.05):
    """
    Bootstrap confidence interval over sample-level values.
    metric_fn: accepts (sample_weights) or (values, weights) and returns a scalar metric.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(values[idx]))
    stats = np.array(stats)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi

# ============ Financial simulation ============
def simulate_bets(y_true, proba, odds_h, odds_d, odds_a, stake=1.0,
                  min_prob=0.0, no_bet_mask=None):
    """
    Simulate betting: for each sample, bet on the outcome with the highest model
    probability (if it exceeds the threshold). Returns per-sample returns.
    """
    n = len(y_true)
    returns = np.zeros(n)
    pred = proba.argmax(axis=1)
    placed = np.zeros(n, dtype=bool)
    for i in range(n):
        if no_bet_mask is not None and no_bet_mask[i]:
            continue
        p_max = proba[i].max()
        if p_max < min_prob:
            continue
        odds = [odds_h[i], odds_d[i], odds_a[i]][pred[i]]
        if not np.isfinite(odds) or odds <= 1:
            continue
        placed[i] = True
        if pred[i] == y_true[i]:
            returns[i] = stake * (odds - 1)
        else:
            returns[i] = -stake
    return returns, placed

def financial_metrics(returns, placed=None):
    """Compute ROI/Sharpe/MDD/win rate from per-sample returns. MDD is the maximum drawdown relative to peak capital."""
    if placed is not None:
        r = returns[placed]
    else:
        r = returns
    if len(r) == 0:
        return {"n_bets": 0, "roi": np.nan, "sharpe": np.nan, "mdd": np.nan, "win_rate": np.nan}
    total_stake = len(r)
    net = r.sum()
    roi = net / total_stake
    sharpe = (r.mean() / r.std()) if r.std() > 0 else np.nan
    cum = np.cumsum(r)
    # Wealth curve: initial capital = total staked (equal-stakes strategy), 1 unit per bet
    wealth = 1 + cum / total_stake
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / peak
    mdd = dd.min()
    win_rate = (r > 0).mean()
    return {"n_bets": int(len(r)), "roi": roi, "sharpe": sharpe, "mdd": mdd, "win_rate": win_rate}

def evaluate_predictions(y_true, proba):
    """Summary of prediction metrics"""
    return {
        "accuracy": accuracy_score(y_true, proba.argmax(axis=1)),
        "macro_f1": f1_score(y_true, proba.argmax(axis=1), average="macro"),
        "log_loss": log_loss(y_true, proba, labels=[0, 1, 2]),
        "brier": brier_multiclass(y_true, proba),
        "ece": ece(y_true, proba),
    }

def class_metrics(y_true, proba, labels=["H", "D", "A"]):
    from sklearn.metrics import precision_recall_fscore_support
    pred = proba.argmax(axis=1)
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2])
    return pd.DataFrame({"class": labels, "precision": p, "recall": r, "f1": f})
