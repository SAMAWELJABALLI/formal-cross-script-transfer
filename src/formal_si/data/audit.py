"""Dataset audit: counts, label balance, script mix, duplicate groups, split leakage.

Usage: python -m formal_si.data.audit --examples data/examples.jsonl --out reports/data_audit.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..text.normalize import arabic_script_ratio
from .schema import Example


def audit(examples: list[Example]) -> dict:
    by_source: dict[str, list[Example]] = defaultdict(list)
    for e in examples:
        by_source[e.source].append(e)
    report = {}
    for src, rows in sorted(by_source.items()):
        texts = [e.text for e in rows]
        group_sizes = Counter(e.group for e in rows)
        split_of_group = defaultdict(set)
        for e in rows:
            split_of_group[e.group].add(e.split)
        report[src] = {
            "n": len(rows),
            "labels": dict(Counter(e.label for e in rows)),
            "forms": dict(Counter(e.form for e in rows)),
            "splits": dict(Counter(e.split for e in rows)),
            "mean_arabic_script_ratio": round(sum(map(arabic_script_ratio, texts)) / len(texts), 4),
            "exact_duplicate_texts": len(texts) - len(set(texts)),
            "groups": len(group_sizes),
            "multi_row_groups": sum(1 for v in group_sizes.values() if v > 1),
            "groups_in_several_splits": sum(1 for v in split_of_group.values() if len(v) > 1),
            "mean_chars": round(sum(map(len, texts)) / len(texts), 1),
        }
    return report


def load_jsonl(path: Path) -> list[Example]:
    with open(path, encoding="utf-8") as f:
        return [Example(**json.loads(line)) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="data/examples.jsonl")
    ap.add_argument("--out", default="reports/data_audit.json")
    args = ap.parse_args()
    rep = audit(load_jsonl(Path(args.examples)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
