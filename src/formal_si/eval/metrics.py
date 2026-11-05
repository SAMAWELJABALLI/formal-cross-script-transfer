"""Evaluation metrics: task quality, cross-form consistency, invariance, sensitivity."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def cross_form_consistency(pred, groups) -> float:
    """Share of groups (one sentence in several forms) whose forms all get the same label.
    Groups with a single row are ignored."""
    by_group = defaultdict(set)
    counts = defaultdict(int)
    for p, g in zip(pred, groups):
        by_group[g].add(p)
        counts[g] += 1
    multi = [g for g in by_group if counts[g] > 1]
    if not multi:
        return float("nan")
    return float(np.mean([len(by_group[g]) == 1 for g in multi]))


def _cos_dist(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)
    b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(1e-12)
    return 1.0 - (a * b).sum(1)


def invariance_score(pair_a, pair_b, seed: int = 0) -> float:
    """1 - mean d(meaning-preserving pair) / mean d(random pair). 1 = perfectly invariant,
    0 = pairs no closer than random sentences."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pair_b))
    d_pair = _cos_dist(pair_a, pair_b).mean()
    d_rand = _cos_dist(pair_a, pair_b[perm]).mean()
    return float(1.0 - d_pair / d_rand) if d_rand > 0 else float("nan")


def sensitivity_auc(anchor, preserved, changed) -> float:
    """AUC that a meaning-changing variant is farther from the anchor than a
    meaning-preserving variant. anchor/preserved/changed are aligned (n, d) arrays."""
    d_pos = _cos_dist(anchor, preserved)
    d_neg = _cos_dist(anchor, changed)
    labels = np.r_[np.zeros(len(d_pos)), np.ones(len(d_neg))]
    return float(roc_auc_score(labels, np.r_[d_pos, d_neg]))


def flip_rate(pred_original, pred_negated) -> float:
    """Share of negated sentences whose predicted sentiment label changed."""
    return float(np.mean(np.asarray(pred_original) != np.asarray(pred_negated)))


def hypervolume_2d(points, ref=(0.0, 0.0)) -> float:
    """Area dominated by a set of (x, y) points, both maximised, above ref."""
    pts = sorted({(float(x), float(y)) for x, y in points if x > ref[0] and y > ref[1]},
                 key=lambda p: -p[0])
    # Sweep by x descending: between consecutive x values the dominated height is the
    # best y seen so far.
    area = 0.0
    best_y = ref[1]
    xs = [p[0] for p in pts] + [ref[0]]
    for i, (x, y) in enumerate(pts):
        best_y = max(best_y, y)
        area += (x - xs[i + 1]) * (best_y - ref[1])
    return float(area)
