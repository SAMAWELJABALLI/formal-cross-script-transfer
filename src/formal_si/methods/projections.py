"""Linear invariance methods applied to frozen, cached sentence embeddings.

Every method follows the same interface: ``fit(...)`` on training data only, then
``transform(X)``. All operate in the original embedding space (output dim = input dim)
so a downstream linear classifier is trained identically for every method.

Implemented:
- FormalSI: pair-specified selective invariance (this project's method, Section 4).
- LEACE: least-squares concept erasure (Belrose et al., 2023), closed form.
- SplinceStyle: oblique projection that erases a concept while preserving the
  covariance with the task label, after Holstege, Ravfogel & Wouters (2025).
  The kernel and range constraints are exact; the choice of residual subspace is a
  simplification of the paper's distortion-minimising choice (documented below).
- MeanCentering: subtract per-group (e.g. per-script) means.
- LSAR-style low-rank removal of the subspace spanned by group means
  (after Xie et al., EMNLP 2022).
- RandomRemoval: removes k random orthonormal directions (control).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _sym_inv_sqrt(cov: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (W, W_pinv) with W = cov^{-1/2} computed on the non-null eigenspace."""
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
    keep = vals > eps * max(vals.max(), 1e-12)
    v, q = vals[keep], vecs[:, keep]
    w = (q / np.sqrt(v)) @ q.T
    w_pinv = (q * np.sqrt(v)) @ q.T
    return w, w_pinv


def _orth(a: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Orthonormal basis for the column space of ``a``."""
    if a.size == 0:
        return np.zeros((a.shape[0], 0))
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    rank = int((s > tol * max(s.max(), 1e-12)).sum())
    return u[:, :rank]


def _one_hot(y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    return (y[:, None] == classes[None, :]).astype(np.float64)


def difference_scatter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """S = mean over pairs of (a_i - b_i)(a_i - b_i)^T, shape (d, d)."""
    if a.shape != b.shape:
        raise ValueError("pair arrays must have identical shapes")
    if len(a) == 0:
        return np.zeros((a.shape[1], a.shape[1]))
    diff = a - b
    return diff.T @ diff / len(diff)


def between_class_scatter(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """S_task = (1/n) sum_c n_c (mu_c - mu)(mu_c - mu)^T."""
    mu = x.mean(0)
    s = np.zeros((x.shape[1], x.shape[1]))
    for c in np.unique(y):
        xc = x[y == c]
        d = (xc.mean(0) - mu)[:, None]
        s += len(xc) * (d @ d.T)
    return s / len(x)


class _Base:
    def fit(self, *args, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError

    def transform(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class Identity(_Base):
    name = "raw"

    def fit(self, *args, **kwargs):
        return self

    def transform(self, x):
        return np.asarray(x, dtype=np.float64)


@dataclass
class FormalSI(_Base):
    """Remove the k directions maximising tr(U^T [S_inv - lam (S_task + beta S_con)] U).

    lam = lam0 * tr(S_inv) / tr(S_task + beta S_con) makes lam0 scale-free.
    The transform is the orthogonal projection h = (I - U U^T) x.
    """

    k: int = 16
    lam0: float = 0.5
    beta: float = 1.0
    name: str = "formal_si"
    U: np.ndarray | None = field(default=None, repr=False)
    eigvals: np.ndarray | None = field(default=None, repr=False)

    def fit(self, x_train, y_train, pair_a, pair_b, con_a=None, con_b=None,
            inv_weights=None):
        """pair_a/pair_b: embeddings of meaning-preserving pairs (n_pairs, d), or a list
        of such (a, b) tuples, one per transformation type, combined with inv_weights."""
        x_train = np.asarray(x_train, dtype=np.float64)
        d = x_train.shape[1]
        if isinstance(pair_a, (list, tuple)):
            weights = inv_weights or [1.0] * len(pair_a)
            s_inv = sum(w * difference_scatter(np.asarray(a, float), np.asarray(b, float))
                        for w, a, b in zip(weights, pair_a, pair_b))
        else:
            s_inv = difference_scatter(np.asarray(pair_a, float), np.asarray(pair_b, float))
        s_prot = between_class_scatter(x_train, np.asarray(y_train))
        if con_a is not None and self.beta > 0 and len(con_a):
            s_prot = s_prot + self.beta * difference_scatter(np.asarray(con_a, float),
                                                             np.asarray(con_b, float))
        tr_prot = np.trace(s_prot)
        lam = self.lam0 * np.trace(s_inv) / tr_prot if tr_prot > 0 else 0.0
        m = s_inv - lam * s_prot
        vals, vecs = np.linalg.eigh((m + m.T) / 2)
        order = np.argsort(vals)[::-1]
        k = min(self.k, d)
        # Only remove directions with positive objective: removing a direction whose
        # protected variance outweighs its invariance variance would hurt by design.
        top = [i for i in order[:k] if vals[i] > 0]
        self.U = vecs[:, top]
        self.eigvals = vals[order]
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        return x - (x @ self.U) @ self.U.T


@dataclass
class LEACE(_Base):
    """r(x) = x - W^+ P_{W Sigma_xz} W (x - mu)."""

    eps: float = 1e-8
    name: str = "leace"

    def fit(self, x, z, *args, **kwargs):
        x = np.asarray(x, dtype=np.float64)
        zc = _one_hot(np.asarray(z)) if np.asarray(z).ndim == 1 else np.asarray(z, float)
        self.mu = x.mean(0)
        xc = x - self.mu
        sxx = xc.T @ xc / len(x)
        sxz = xc.T @ (zc - zc.mean(0)) / len(x)
        w, w_pinv = _sym_inv_sqrt(sxx, self.eps)
        b = _orth(w @ sxz)
        self.M = w_pinv @ (b @ b.T) @ w
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        return x - (x - self.mu) @ self.M.T


@dataclass
class SplinceStyle(_Base):
    """Oblique projection Q = V (U^T V)^{-1} U^T in whitened space.

    U spans colsp(W Sigma_xz)^perp (kernel constraint: Q W Sigma_xz = 0).
    V = [orth(W Sigma_xy), leading directions of (I - A A^T) U]  (range constraint:
    Q W Sigma_xy = W Sigma_xy). The paper picks the residual part of V to minimise
    distortion; we take the projection of U off the task directions, which satisfies
    both constraints exactly but is not guaranteed distortion-optimal.
    """

    eps: float = 1e-8
    name: str = "splince_style"

    def fit(self, x, z, y, *args, **kwargs):
        x = np.asarray(x, dtype=np.float64)
        zc = _one_hot(np.asarray(z))
        yc = _one_hot(np.asarray(y))
        self.mu = x.mean(0)
        xc = x - self.mu
        n = len(x)
        sxx = xc.T @ xc / n
        sxz = xc.T @ (zc - zc.mean(0)) / n
        sxy = xc.T @ (yc - yc.mean(0)) / n
        w, w_pinv = _sym_inv_sqrt(sxx, self.eps)
        d = x.shape[1]
        b = _orth(w @ sxz)
        u = _orth(np.eye(d) - b @ b.T)
        a = _orth(w @ sxy)
        # The oblique projection can keep task directions that are correlated with (but
        # not inside) the concept span; that is exactly where it differs from LEACE.
        resid = _orth((np.eye(d) - a @ a.T) @ u)
        need = u.shape[1] - a.shape[1]
        v = np.concatenate([a, resid[:, :max(need, 0)]], axis=1)
        q = v @ np.linalg.pinv(u.T @ v) @ u.T
        self.M = w_pinv @ q @ w
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        return self.mu + (x - self.mu) @ self.M.T


@dataclass
class MeanCentering(_Base):
    """Subtract the training mean of each group (e.g. script); unseen groups use the
    global mean."""

    name: str = "mean_centering"

    def fit(self, x, groups, *args, **kwargs):
        x = np.asarray(x, dtype=np.float64)
        groups = np.asarray(groups)
        self.global_mean = x.mean(0)
        self.means = {g: x[groups == g].mean(0) for g in np.unique(groups)}
        return self

    def transform(self, x, groups=None):
        x = np.asarray(x, dtype=np.float64)
        if groups is None:
            raise ValueError("MeanCentering.transform needs the group of each row")
        mu = np.stack([self.means.get(g, self.global_mean) for g in np.asarray(groups)])
        return x - mu + self.global_mean


@dataclass
class LowRankMeanRemoval(_Base):
    """LSAR-style: remove the top-r singular directions of the centred group-mean matrix."""

    r: int = 1
    name: str = "lsar_style"

    def fit(self, x, groups, *args, **kwargs):
        x = np.asarray(x, dtype=np.float64)
        groups = np.asarray(groups)
        means = np.stack([x[groups == g].mean(0) for g in np.unique(groups)])
        means = means - means.mean(0)
        _, _, vt = np.linalg.svd(means, full_matrices=False)
        self.U = vt[: min(self.r, len(vt))].T
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        return x - (x @ self.U) @ self.U.T


@dataclass
class RandomRemoval(_Base):
    k: int = 16
    seed: int = 0
    name: str = "random_removal"

    def fit(self, x, *args, **kwargs):
        d = np.asarray(x).shape[1]
        rng = np.random.default_rng(self.seed)
        self.U = _orth(rng.standard_normal((d, min(self.k, d))))
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        return x - (x @ self.U) @ self.U.T
