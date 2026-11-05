"""Mathematical properties of the projection methods, on synthetic Gaussian data.
These verify the implementation, not any research claim."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from formal_si.methods.projections import (FormalSI, LEACE, LowRankMeanRemoval,
                                           MeanCentering, RandomRemoval, SplinceStyle,
                                           between_class_scatter, difference_scatter)


def synthetic(n=600, d=20, seed=0):
    """Dim 0 carries the label, dim 1 carries the 'script', dims 2.. are noise.
    Pairs differ only along dim 1 (+ small noise)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    z = rng.integers(0, 2, n)
    x = rng.normal(0, 0.3, (n, d))
    x[:, 0] += 2.0 * y
    x[:, 1] += 3.0 * z
    a = x.copy()
    b = x.copy()
    b[:, 1] += 3.0 + rng.normal(0, 0.05, n)  # same sentence, other script
    return x, y, z, a, b


def test_scatter_shapes_and_psd():
    x, y, _, a, b = synthetic()
    for s in (difference_scatter(a, b), between_class_scatter(x, y)):
        assert s.shape == (20, 20)
        assert np.allclose(s, s.T)
        assert np.linalg.eigvalsh(s).min() > -1e-9


def test_formal_si_removes_pair_direction_and_keeps_label_direction():
    x, y, _, a, b = synthetic()
    m = FormalSI(k=1, lam0=1.0, beta=0.0).fit(x, y, a, b)
    u = m.U[:, 0]
    assert abs(u[1]) > 0.99          # removed the 'script' axis
    h = m.transform(x)
    assert np.allclose(h @ u, 0, atol=1e-10)
    assert np.allclose(m.transform(h), h)  # idempotent projection
    assert np.corrcoef(h[:, 0], y)[0, 1] > 0.9  # label axis untouched


def test_formal_si_does_not_remove_direction_dominated_by_task():
    # When the pair direction coincides with the label direction and lam is large,
    # the objective is negative there and nothing is removed.
    rng = np.random.default_rng(1)
    n, d = 400, 6
    y = rng.integers(0, 2, n)
    x = rng.normal(0, 0.1, (n, d))
    x[:, 0] += 3 * y
    a, b = x.copy(), x.copy()
    b[:, 0] += 0.2
    m = FormalSI(k=3, lam0=50.0, beta=0.0).fit(x, y, a, b)
    assert all(abs(m.U[0, j]) < 0.5 for j in range(m.U.shape[1]))


def test_formal_si_contrast_term_protects_negation_direction():
    x, y, _, a, b = synthetic()
    # Pairs vary along dims 1 and 2; contrast (negation) pairs vary along dim 2.
    rng = np.random.default_rng(2)
    b2 = b.copy()
    b2[:, 2] += rng.normal(0, 2.0, len(b))
    ca, cb = x.copy(), x.copy()
    cb[:, 2] += 3.0
    no = FormalSI(k=2, lam0=0.0, beta=0.0).fit(x, y, a, b2)
    yes = FormalSI(k=2, lam0=1.0, beta=5.0).fit(x, y, a, b2, ca, cb)
    removed_no = np.abs(no.U[2]).max()
    removed_yes = np.abs(yes.U[2]).max() if yes.U.size else 0.0
    assert removed_no > 0.9 and removed_yes < 0.1


def test_leace_removes_linear_information():
    x, y, z, *_ = synthetic()
    h = LEACE().fit(x, z).transform(x)
    zc = z - z.mean()
    cov = (h - h.mean(0)).T @ zc / len(z)
    assert np.abs(cov).max() < 1e-8
    acc = LogisticRegression(max_iter=1000).fit(h, z).score(h, z)
    assert acc < 0.6  # near chance after erasure


def test_splince_style_constraints():
    x, y, z, *_ = synthetic()
    m = SplinceStyle().fit(x, z, y)
    h = m.transform(x)
    zc, yc = z - z.mean(), y - y.mean()
    assert np.abs((h - h.mean(0)).T @ zc / len(z)).max() < 1e-8
    cov_before = (x - x.mean(0)).T @ yc / len(y)
    cov_after = (h - h.mean(0)).T @ yc / len(y)
    assert np.allclose(cov_before, cov_after, atol=1e-8)


def test_centering_and_lsar_and_random():
    x, y, z, *_ = synthetic()
    groups = np.where(z == 1, "arabic", "arabizi")
    h = MeanCentering().fit(x, groups).transform(x, groups)
    assert abs(h[z == 1, 1].mean() - h[z == 0, 1].mean()) < 1e-8
    l = LowRankMeanRemoval(r=1).fit(x, groups)
    assert abs(l.U[1, 0]) > 0.95
    r = RandomRemoval(k=3, seed=0).fit(x)
    assert np.allclose(r.U.T @ r.U, np.eye(3), atol=1e-10)


def test_pair_arrays_must_match():
    with pytest.raises(ValueError):
        difference_scatter(np.zeros((3, 2)), np.zeros((4, 2)))
