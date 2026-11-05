"""Build the negation contrast set from human cross-script pairs (NArabizi, TArC).

Each row: {text: anchor, paraphrase: the same sentence in the other script (human),
negated: the anchor with a rule-based Maghrebi negation, split, source, group, direction}.
The anchor is the normalised form of the human sentence, so anchor and negated differ
only by the negation edit. Paraphrases are the untouched human transliteration.

Train rows feed the beta * S_con term of FORMAL-SI; dev rows measure negation AUC and
flip rate. Because both come from the same rule, negation AUC under beta > 0 is
in-distribution for the rule and is reported as such. Test rows are not built unless
--splits includes test (only after a go decision).

The negations are rule-generated and unvalidated: a native speaker must check a
100-sentence sample before any number from this set is published.

Usage: python -m formal_si.data.contrast --out data/contrast_negation.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..text.normalize import negate_arabic, negate_arabizi, normalize_arabic, normalize_arabizi
from .audit import load_jsonl

ARABIC_FORM = {"narabizi": "arabic", "tarc": "codeswitch"}


def build(examples, splits=("train", "dev")):
    groups = defaultdict(dict)
    for e in examples:
        if e.source in ARABIC_FORM and e.split in splits:
            groups[(e.source, e.group, e.split)].setdefault(e.form, e.text)
    rows, stats = [], Counter()
    for (source, group, split), forms in sorted(groups.items()):
        az, ar = forms.get("arabizi"), forms.get(ARABIC_FORM[source])
        if not az or not ar:
            continue
        for direction, anchor, para, neg_fn, norm in (
                ("arabizi", az, ar, negate_arabizi, normalize_arabizi),
                ("arabic", ar, az, negate_arabic, normalize_arabic)):
            neg = neg_fn(anchor)
            stats[(split, direction, "skipped" if neg is None else "built")] += 1
            if neg is None or neg == norm(anchor):
                continue
            rows.append({"text": norm(anchor), "paraphrase": para, "negated": neg,
                         "split": split, "source": source, "group": group,
                         "direction": direction, "validated": False})
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="data/examples.jsonl")
    ap.add_argument("--out", default="data/contrast_negation.jsonl")
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    args = ap.parse_args()
    rows, stats = build(load_jsonl(Path(args.examples)), tuple(args.splits))
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"|".join(k): v for k, v in sorted(stats.items())}, indent=1))
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
