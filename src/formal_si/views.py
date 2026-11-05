"""Input views shared by encode.py and run.py, so the cache and the experiment always see
the same strings.

- Emoji are stripped for the sources listed in ``strip_emoji_sources`` (MYC: emoji carry
  sentiment independently of script and confound cross-script evaluation).
- ``deromanized`` maps every non-Arabic-script row through the rule deromaniser and
  normalises Arabic-script rows, exactly as the encoder-free baseline does. It is the
  deromanize-then-encode condition (TextPro-SLM's input-side principle applied to text).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .text.normalize import deromanize, detect_script, normalize_arabic, strip_emoji

VIEWS = ("raw", "deromanized")


def view_text(text: str, form: str, view: str) -> str:
    if view == "raw":
        return text
    if view == "deromanized":
        return normalize_arabic(text) if form == "arabic" else normalize_arabic(deromanize(text))
    raise ValueError(view)


def form_of_text(text: str) -> str:
    """Form label for a free string (contrast rows carry no form)."""
    return {"arab": "arabic", "latn": "arabizi"}.get(detect_script(text), "mixed")


def apply_emoji(examples, sources):
    sources = set(sources or [])
    out = []
    for e in examples:
        if e.source in sources:
            e = copy.copy(e)
            e.text = strip_emoji(e.text)
        out.append(e)
    return out


def apply_view(examples, view):
    if view == "raw":
        return examples
    out = []
    for e in examples:
        e = copy.copy(e)
        e.text = view_text(e.text, e.form, view)
        out.append(e)
    return out


def contrast_texts(path) -> list[str]:
    if not path or not Path(path).exists():
        return []
    ts = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            ts += [t for t in (r.get("text"), r.get("paraphrase"), r.get("negated")) if t]
    return ts


def all_texts(cfg: dict, examples) -> list[str]:
    """Every string any view of the pilot will look up, for one pass of encoding.
    Test rows are excluded unless evaluate_split is test: they are never needed."""
    splits = {"train", cfg.get("evaluate_split", "dev")}
    ex = [e for e in apply_emoji(examples, cfg.get("strip_emoji_sources")) if e.split in splits]
    texts = set()
    for view in cfg.get("input_views", ["raw"]):
        texts |= {e.text for e in apply_view(ex, view)}
        texts |= {view_text(t, form_of_text(t), view)
                  for t in contrast_texts(cfg.get("contrast_file"))}
    return sorted(texts)
