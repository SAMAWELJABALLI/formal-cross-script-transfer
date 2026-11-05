import numpy as np
import pytest

from formal_si.data.schema import Example, check_no_leakage, grouped_split, merge_groups
from formal_si.eval import metrics as M
from formal_si.eval.stats import grouped_paired_bootstrap, holm
from formal_si.methods.adapter import AdapterTrainer
from formal_si.text import normalize as N


def test_consistency_and_flip():
    assert M.cross_form_consistency(["p", "p", "n", "p"], ["a", "a", "b", "b"]) == 0.5
    assert M.flip_rate(["p", "n"], ["n", "n"]) == 0.5


def test_invariance_and_sensitivity():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 16))
    assert M.invariance_score(a, a + 1e-3 * rng.normal(size=a.shape)) > 0.99
    neg = a + rng.normal(size=a.shape)
    para = a + 0.05 * rng.normal(size=a.shape)
    assert M.sensitivity_auc(a, para, neg) > 0.95


def test_hypervolume():
    assert M.hypervolume_2d([(1, 1)]) == pytest.approx(1.0)
    assert M.hypervolume_2d([(1, 0.5), (0.5, 1)]) == pytest.approx(0.75)
    assert M.hypervolume_2d([(1, 1), (0.5, 0.5)]) == pytest.approx(1.0)


def test_bootstrap_resamples_groups():
    y = np.array([0, 0, 1, 1] * 50)
    groups = np.repeat(np.arange(100), 2)
    a = y.copy()
    b = y.copy()
    b[::4] = 1 - b[::4]
    res = grouped_paired_bootstrap(y, a, b, groups, M.macro_f1, n_boot=500)
    assert res["delta"] > 0 and res["ci_low"] > 0 and res["n_groups"] == 100


def test_holm():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out["a"]["p_holm"] == pytest.approx(0.03)
    assert out["c"]["p_holm"] == pytest.approx(0.06)
    assert out["b"]["p_holm"] == pytest.approx(0.06)
    assert out["a"]["reject"] and not out["b"]["reject"]


def _ex(uid, text, group, split=None):
    return Example(uid=uid, text=text, group=group, form="arabizi", dialect="TN",
                   source="s", task="sentiment", label="pos", split=split)


def test_near_duplicates_merged_and_split_without_leakage():
    exs = [_ex("1", "ya3tik sa7a 3la had el video", "a"),
           _ex("2", "ya3tik sa7a 3la had el video!!", "b"),
           _ex("3", "matchou khayeb barcha", "c"),
           _ex("4", "يعطيك الصحة على هذا الفيديو", "a")]
    merge_groups(exs, threshold=0.6)
    assert exs[0].group == exs[1].group == exs[3].group
    assert exs[2].group != exs[0].group
    exs[3].split = "test"
    grouped_split(exs, seed=0)
    check_no_leakage(exs)
    assert exs[0].split == "test"


def test_script_and_normalization():
    assert N.detect_script("نحبك برشا") == "arab"
    assert N.detect_script("n7ebek barcha") == "latn"
    assert N.normalize_arabizi("BAAARCHA   mezyen") == "baarcha mezyen"
    assert N.deromanize("3ala") == "عالا"
    assert N.negate_arabizi("n7eb el film") == "man7ebch el film"
    assert N.negate_arabizi("man7ebch") is None
    assert N.negate_arabic("نحب الفيلم") == "مانحبش الفيلم"


def test_adapter_learns_and_respects_hinge():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 12)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int)
    t = AdapterTrainer(rank=4, epochs=30, alpha=0.5, gamma=1.0, lr=1e-2, seed=0)
    t.fit(x, y, x, x + 0.01, x[:50], x[:50] + 1.0)
    assert t.history[-1] < t.history[0]
    assert (t.predict(x) == y).mean() > 0.9
    assert t.model.n_adapter_params() == 2 * 12 * 4
    gated = AdapterTrainer(rank=4, epochs=2, gated=True, seed=0).fit(x, y, n_tasks=2, task=1)
    assert gated.model.n_adapter_params() == 2 * 12 * 4 + 2 * 12


def test_strip_emoji():
    assert N.strip_emoji("nhebek ❤️ barcha 😂") == "nhebek barcha"
