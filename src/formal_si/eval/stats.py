"""Paired, group-level bootstrap and multiple-comparison correction."""

from __future__ import annotations

from typing import Callable

import numpy as np


def grouped_paired_bootstrap(y_true, pred_a, pred_b, groups,
                             metric: Callable, n_boot: int = 10_000, seed: int = 0,
                             alpha: float = 0.05) -> dict:
    """Bootstrap the difference metric(a) - metric(b), resampling whole groups so that
    all forms of one sentence move together (the correct unit when forms are paired).

    Returns the observed delta, the percentile CI, and a two-sided bootstrap p-value
    (share of resampled deltas on the other side of zero, doubled)."""
    y_true, pred_a, pred_b, groups = map(np.asarray, (y_true, pred_a, pred_b, groups))
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_by_group = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    observed = metric(y_true, pred_a) - metric(y_true, pred_b)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_group[g] for g in pick])
        deltas[i] = metric(y_true[idx], pred_a[idx]) - metric(y_true[idx], pred_b[idx])
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {"delta": float(observed), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(p, 1.0)), "n_groups": int(len(uniq)), "n_boot": n_boot}


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down correction. Returns adjusted p and reject flag."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, max(running, (m - rank) * p))
        running = adj
        out[name] = {"p": p, "p_holm": adj, "reject": adj < alpha}
    return out
