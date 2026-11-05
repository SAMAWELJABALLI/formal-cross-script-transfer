"""Input views, emoji stripping and the contrast builder (synthetic strings only)."""

from formal_si.data.contrast import build
from formal_si.data.schema import Example
from formal_si.views import all_texts, apply_emoji, apply_view, view_text


def _ex(text, form, group, split="train", source="narabizi"):
    return Example(uid=f"{group}:{form}", text=text, group=group, form=form, dialect="DZ",
                   source=source, task="sentiment", label="POS", split=split)


def test_views_and_emoji():
    ex = [_ex("rani nhabek 😍", "arabizi", "g1", source="myc"),
          _ex("راني نحبك", "arabic", "g1", source="myc")]
    stripped = apply_emoji(ex, ["myc"])
    assert "😍" not in stripped[0].text and "😍" in ex[0].text  # originals untouched
    derom = apply_view(stripped, "deromanized")
    assert all(not any("a" <= c <= "z" for c in e.text) for e in derom)
    assert view_text("abc", "arabizi", "raw") == "abc"


def test_contrast_and_all_texts(tmp_path):
    ex = [_ex("rani nhabek bezzaf", "arabizi", "g1"), _ex("راني نحبك بزاف", "arabic", "g1"),
          _ex("x", "arabizi", "g2", split="test"), _ex("ي", "arabic", "g2", split="test")]
    rows, _ = build(ex)
    assert rows and all(r["split"] == "train" for r in rows)  # test never built by default
    assert {r["direction"] for r in rows} == {"arabizi", "arabic"}
    assert all(r["negated"] != r["text"] for r in rows)
    cfg = {"evaluate_split": "dev", "input_views": ["raw", "deromanized"]}
    texts = all_texts(cfg, ex)
    assert "x" not in texts and "rani nhabek bezzaf" in texts
