"""Data-driven Arabizi -> Arabic-script token transliterator (stronger-transliterator control).

A token dictionary learned from human-aligned sentence pairs, with the rule deromaniser as fallback:
  * TArC train: Arabizi layer vs code-switched layer (Arabic-script tokens only), token-count-equal sentences;
  * NArabizi train: Arabizi vs Arabic-script layer, token-count-equal sentences.
For each normalised Arabizi token the most frequent Arabic-script token is kept. Alignment is positional and
only used when both sentences have the same number of tokens.

``fit(pairs)`` / ``transliterate(text)``; ``cross_fitted(...)`` gives, for each NArabizi train group, the
transliteration by a dictionary that never saw that group (2 folds by group id hash).
Usage (quality report): python -m formal_si.translit --out reports/confirm/translit_quality.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .text.normalize import _is_arabic, deromanize, normalize_arabic, normalize_arabizi


def _norm_tok(t: str) -> str:
    return normalize_arabizi(t).strip(".,!?;:\"'()")


def aligned_token_pairs(pairs):
    """pairs: (arabizi_sentence, arabic_script_sentence)."""
    for az, ar in pairs:
        ta, tr = az.split(), normalize_arabic(ar).split()
        if len(ta) != len(tr):
            continue
        for a, r in zip(ta, tr):
            a = _norm_tok(a)
            if a and any(_is_arabic(c) for c in r) and not any(_is_arabic(c) for c in a):
                yield a, r


class DictTransliterator:
    def __init__(self):
        self.table: dict[str, str] = {}

    def fit(self, pairs):
        counts = defaultdict(Counter)
        for a, r in aligned_token_pairs(pairs):
            counts[a][r] += 1
        self.table = {a: c.most_common(1)[0][0] for a, c in counts.items()}
        return self

    def transliterate(self, text: str) -> str:
        out = []
        for tok in normalize_arabizi(text).split():
            key = _norm_tok(tok)
            out.append(self.table.get(key) or normalize_arabic(deromanize(tok)))
        return " ".join(t for t in out if t)


def fold_of(group: str) -> int:
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % 2


def resources(examples):
    """(tarc_pairs, narabizi_groups) from TRAIN rows only."""
    tr = [e for e in examples if e.split == "train"]
    g = defaultdict(dict)
    for e in tr:
        if e.source in ("tarc", "narabizi"):
            g[(e.source, e.group)].setdefault(e.form, e.text)
    tarc = [(fm["arabizi"], fm["codeswitch"]) for (s, _), fm in g.items()
            if s == "tarc" and "arabizi" in fm and "codeswitch" in fm]
    nar = {grp: fm for (s, grp), fm in g.items() if s == "narabizi" and "arabizi" in fm and "arabic" in fm}
    return tarc, nar


def full_dictionary(examples) -> DictTransliterator:
    tarc, nar = resources(examples)
    return DictTransliterator().fit(tarc + [(fm["arabizi"], fm["arabic"]) for fm in nar.values()])


def cross_fitted(examples) -> dict[str, str]:
    """group -> dictionary transliteration of its Arabizi sentence, by the other fold's dictionary."""
    tarc, nar = resources(examples)
    out = {}
    for fold in (0, 1):
        train_pairs = tarc + [(fm["arabizi"], fm["arabic"]) for grp, fm in nar.items() if fold_of(grp) != fold]
        d = DictTransliterator().fit(train_pairs)
        for grp, fm in nar.items():
            if fold_of(grp) == fold:
                out[grp] = d.transliterate(fm["arabizi"])
    return out


def cer(a: str, b: str) -> float:
    """Character error rate of a against reference b (Levenshtein / len(b))."""
    a, b = a.replace(" ", ""), b.replace(" ", "")
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(b))


def main():
    from .data.audit import load_jsonl
    from .views import apply_emoji
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="data/examples.jsonl")
    ap.add_argument("--out", default="reports/confirm/translit_quality.json")
    args = ap.parse_args()
    ex = [e for e in apply_emoji(load_jsonl(Path(args.examples)), ["myc", "tsac", "tunizi", "narabizi", "tarc"])
          if e.split in ("train", "dev")]
    _, nar = resources(ex)
    cf = cross_fitted(ex)
    full = full_dictionary(ex)
    dev = defaultdict(dict)
    for e in ex:
        if e.split == "dev" and e.source == "narabizi":
            dev[e.group].setdefault(e.form, e.text)
    dev = [fm for fm in dev.values() if "arabizi" in fm and "arabic" in fm]

    def mean_cer(pairs):
        return sum(cer(h, normalize_arabic(r)) for h, r in pairs) / len(pairs)

    rep = {
        "dictionary_size_full": len(full.table),
        "train_cross_fitted_cer_rule": mean_cer([(normalize_arabic(deromanize(fm["arabizi"])), fm["arabic"]) for fm in nar.values()]),
        "train_cross_fitted_cer_dict": mean_cer([(cf[g], fm["arabic"]) for g, fm in nar.items()]),
        "dev_cer_rule": mean_cer([(normalize_arabic(deromanize(fm["arabizi"])), fm["arabic"]) for fm in dev]),
        "dev_cer_dict_full": mean_cer([(full.transliterate(fm["arabizi"]), fm["arabic"]) for fm in dev]),
        "n_train_groups": len(nar), "n_dev_groups": len(dev),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
