"""Race simulation shared by every model: turn XGBoost scores into per-car chances.

Each simulated race:
  1. each car fails to finish with its DNF probability (those cars fill the back);
  2. the cars still running are ordered by ranker score + Gumbel noise.
With XGBoost's rank:pairwise, P(i ahead of j) = sigmoid(score_i - score_j), and adding
Gumbel noise to the scores reproduces exactly that, so the noise scale ("temperature")
has a natural value of 1. It's fitted on out-of-sample races to correct over- or
under-confidence.
"""
import numpy as np
import pandas as pd

TEMPS = np.round(np.arange(0.5, 3.01, 0.1), 2)
MIN_TEMP_RACES = 3   # out-of-sample races needed before the temperature is fitted
PROB_CLIP = 0.005


def simulate(scores, p_dnf, n_sims, temperature, rng):
    """Finishing positions (n_sims x cars) and who failed to finish."""
    n = len(scores)
    out = rng.random((n_sims, n)) < p_dnf
    util = scores + temperature * rng.gumbel(size=(n_sims, n))
    util = np.where(out, -1e9 + rng.random((n_sims, n)), util)  # DNFs fill the back, in random order
    order = np.argsort(-util, axis=1)
    finish = np.empty_like(order)
    np.put_along_axis(finish, order, np.broadcast_to(np.arange(1, n + 1), (n_sims, n)), axis=1)
    return finish, out


def summarise(finish, out):
    """Per-car chances from a batch of simulated races."""
    return pd.DataFrame({
        "exp_finish": finish.mean(0), "p_win": (finish == 1).mean(0), "p_top3": (finish <= 3).mean(0),
        "p_top5": (finish <= 5).mean(0), "p_top10": (finish <= 10).mean(0), "p_dnf": out.mean(0),
        "finish_p10": np.percentile(finish, 10, axis=0), "finish_p90": np.percentile(finish, 90, axis=0),
    })


def plackett_luce_loglik(scores, finish, running, temperature):
    """Log-likelihood of the observed order of cars still running, given scores / temperature."""
    s = scores[running] / temperature
    s = s[np.argsort(finish[running])]
    tail = np.logaddexp.accumulate(s[::-1])[::-1]  # log-sum-exp of each car and everyone behind it
    return float((s - tail).sum())


def fit_temperature(oof):
    """Temperature that best explains the finishing orders of out-of-sample races.

    `oof` needs race_id, score, finish_position and dnf columns.
    """
    if oof.empty or oof["race_id"].nunique() < MIN_TEMP_RACES:
        return 1.0
    ll = {t: sum(plackett_luce_loglik(g["score"].to_numpy(), g["finish_position"].to_numpy(),
                                      g["dnf"].to_numpy() == 0, t)
                 for _, g in oof.groupby("race_id")) for t in TEMPS}
    return float(max(ll, key=ll.get))


def logloss(p, y):
    p = np.clip(p, PROB_CLIP, 1 - PROB_CLIP)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def calibration(bt, col="p_top10", target="top10", bins=(0, .05, .1, .2, .3, .4, .5, .6, .8, 1.0)):
    """Predicted vs actual frequency, binned. A calibrated model sits on the diagonal."""
    b = pd.cut(bt[col], bins, include_lowest=True)
    return (bt.groupby(b, observed=True)
            .agg(predicted=(col, "mean"), actual=(target, "mean"), cars=(target, "size"))
            .reset_index(drop=True))
