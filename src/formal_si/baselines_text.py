"""Encoder-free baselines on raw text: character n-gram TF-IDF + logistic regression,
with optional input canonicalisation (normalisation, rule deromanisation).

These are the "simple raw-text and normalized-text systems" of Section 5 and need no
pretrained model. Dev split only unless --split test is passed.

Usage: python -m formal_si.baselines_text --out runs/text_baselines
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data.audit import load_jsonl
from .eval.metrics import cross_form_consistency, macro_f1
from .text.normalize import deromanize, normalize_arabic, normalize_arabizi, strip_emoji

PREP = {
    "raw": lambda t, form: t,
    "normalized": lambda t, form: normalize_arabic(t) if form == "arabic" else normalize_arabizi(t),
    # Canonicalise Latin-script input into Arabic script before featurising (the text
    # analogue of TextPro-SLM's input-side principle). Applied to every non-Arabic row.
    "deromanized": lambda t, form: normalize_arabic(t) if form == "arabic"
    else normalize_arabic(deromanize(t)),
}

# (name, task, sources, train forms, {eval name: forms})
CONDITIONS = [
    ("narabizi_sent_ar2az", "sentiment", ["narabizi"], ["arabic"],
     {"in_script": ["arabic"], "cross_script": ["arabizi"], "codeswitch": ["codeswitch"]}),
    ("narabizi_sent_az2ar", "sentiment", ["narabizi"], ["arabizi"],
     {"in_script": ["arabizi"], "cross_script": ["arabic"], "codeswitch": ["codeswitch"]}),
    ("narabizi_topic_ar2az", "topic", ["narabizi"], ["arabic"],
     {"in_script": ["arabic"], "cross_script": ["arabizi"], "codeswitch": ["codeswitch"]}),
    ("tsac_sent_ar2az", "sentiment", ["tsac"], ["arabic"],
     {"in_script": ["arabic"], "cross_script": ["arabizi"]}),
    ("myc_sent_ar2az", "sentiment", ["myc"], ["arabic"],
     {"in_script": ["arabic"], "cross_script": ["arabizi"]}),
]


def run(examples, out: Path, split: str = "dev", resamples: int = 5, seed: int = 0,
        no_emoji: bool = False):
    out.mkdir(parents=True, exist_ok=True)
    if no_emoji:
        for e in examples:
            e.text = strip_emoji(e.text)
    rows = []
    rng = np.random.default_rng(seed)
    for name, task, sources, train_forms, evals in CONDITIONS:
        data = [e for e in examples if e.task == task and e.source in sources and e.label]
        train = [e for e in data if e.split == "train" and e.form in train_forms]
        ev = {k: [e for e in data if e.split == split and e.form in f] for k, f in evals.items()}
        for prep_name, prep in PREP.items():
            for r in range(resamples):
                s = int(rng.integers(1 << 31))
                sub = np.random.default_rng(s).choice(len(train), int(0.9 * len(train)),
                                                      replace=False)
                tr = [train[i] for i in sub]
                t0 = time.time()
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2,
                                      sublinear_tf=True)
                x = vec.fit_transform([prep(e.text, e.form) for e in tr])
                clf = LogisticRegression(C=1.0, max_iter=3000).fit(x, [e.label for e in tr])
                res = {"condition": name, "prep": prep_name, "resample": r, "seed": s,
                       "n_train": len(tr), "split": split}
                for k, rows_k in ev.items():
                    if not rows_k:
                        continue
                    pred = clf.predict(vec.transform([prep(e.text, e.form) for e in rows_k]))
                    res[f"f1_{k}"] = macro_f1([e.label for e in rows_k], pred)
                    res[f"n_{k}"] = len(rows_k)
                    res[f"consistency_{k}"] = cross_form_consistency(pred, [e.group for e in rows_k])
                res["seconds"] = round(time.time() - t0, 2)
                rows.append(res)
    with open(out / "metrics.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def summary_table(rows) -> str:
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, v in r.items():
            if k.startswith("f1_"):
                agg[(r["condition"], r["prep"])][k].append(100 * v)
    cols = ["f1_in_script", "f1_cross_script", "f1_codeswitch"]
    lines = ["| condition | input | " + " | ".join(c[3:] + " macro-F1" for c in cols) + " | gap (in - cross) |",
             "| --- | --- | " + " | ".join("---:" for _ in cols) + " | ---: |"]
    for (cond, prep), m in agg.items():
        cells = [f"{np.mean(m[c]):.1f} ± {np.std(m[c], ddof=1):.1f}" if m.get(c) else "" for c in cols]
        gap = np.mean(m["f1_in_script"]) - np.mean(m["f1_cross_script"])
        lines.append(f"| {cond} | {prep} | " + " | ".join(cells) + f" | {gap:.1f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="data/examples.jsonl")
    ap.add_argument("--out", default="runs/text_baselines")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--resamples", type=int, default=5)
    ap.add_argument("--no-emoji", action="store_true", help="strip emoji from every text")
    args = ap.parse_args()
    rows = run(load_jsonl(Path(args.examples)), Path(args.out), args.split, args.resamples,
               no_emoji=args.no_emoji)
    table = summary_table(rows)
    (Path(args.out) / "summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
