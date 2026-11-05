"""Common record format and the grouped-split / near-duplicate logic."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Example:
    uid: str          # unique row id
    text: str
    group: str        # all forms / near-duplicates of one sentence share a group
    form: str         # "arabizi" | "arabic" | "codeswitch" | "mixed"
    dialect: str      # "TN" | "DZ" | "MA" | ...
    source: str       # dataset name
    task: str         # "sentiment" | "topic" | ...
    label: str | None
    split: str | None = None   # "train" | "dev" | "test" or None (assigned later)

    def to_dict(self):
        return asdict(self)


def char_ngrams(text: str, n: int = 5) -> set[str]:
    t = f" {text.lower()} "
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def near_duplicate_groups(texts: list[str], threshold: float = 0.8, n: int = 5,
                          band: int = 4) -> list[int]:
    """Cluster texts whose char-n-gram Jaccard similarity exceeds ``threshold``.

    Candidate pairs come from blocking on the ``band`` lexicographically smallest
    n-gram hashes (a cheap MinHash-like signature), then exact Jaccard is checked."""
    grams = [char_ngrams(t, n) for t in texts]
    uf = _UnionFind(len(texts))
    buckets: dict[str, list[int]] = {}
    for i, g in enumerate(grams):
        sig = sorted(hashlib.md5(x.encode()).hexdigest()[:12] for x in g)[:band]
        for s in sig:
            buckets.setdefault(s, []).append(i)
    for members in buckets.values():
        if len(members) > 2000:  # very common n-gram; not informative
            continue
        for a_i in range(len(members)):
            a = members[a_i]
            for b in members[a_i + 1:]:
                if uf.find(a) == uf.find(b):
                    continue
                inter = len(grams[a] & grams[b])
                union = len(grams[a] | grams[b])
                if union and inter / union >= threshold:
                    uf.union(a, b)
    return [uf.find(i) for i in range(len(texts))]


def merge_groups(examples: list[Example], threshold: float = 0.8) -> list[Example]:
    """Merge declared groups with near-duplicate clusters, so a split by group cannot
    leak a near-copy of a test comment into training."""
    roots = near_duplicate_groups([e.text for e in examples], threshold)
    uf = _UnionFind(len(examples))
    first_of_group: dict[str, int] = {}
    first_of_root: dict[int, int] = {}
    for i, (e, r) in enumerate(zip(examples, roots)):
        if e.group in first_of_group:
            uf.union(first_of_group[e.group], i)
        else:
            first_of_group[e.group] = i
        if r in first_of_root:
            uf.union(first_of_root[r], i)
        else:
            first_of_root[r] = i
    for i, e in enumerate(examples):
        e.group = f"g{uf.find(i)}"
    return examples


def grouped_split(examples: list[Example], dev: float = 0.1, test: float = 0.1,
                  seed: int = 0, keep_existing: bool = True) -> list[Example]:
    """Assign splits by group. If ``keep_existing``, a group that already contains a
    dev/test row is placed entirely in that split (dev/test take priority over train)."""
    groups = sorted({e.group for e in examples})
    preset: dict[str, str] = {}
    if keep_existing:
        priority = {"test": 2, "dev": 1, "train": 0}
        for e in examples:
            if e.split in priority:
                cur = preset.get(e.group)
                if cur is None or priority[e.split] > priority[cur]:
                    preset[e.group] = e.split
    rng = np.random.default_rng(seed)
    free = [g for g in groups if g not in preset]
    rng.shuffle(free)
    n_test, n_dev = int(round(test * len(free))), int(round(dev * len(free)))
    assign = dict(preset)
    for g in free[:n_test]:
        assign[g] = "test"
    for g in free[n_test:n_test + n_dev]:
        assign[g] = "dev"
    for g in free[n_test + n_dev:]:
        assign[g] = "train"
    for e in examples:
        e.split = assign[e.group]
    return examples


def check_no_leakage(examples: list[Example]) -> None:
    seen: dict[str, str] = {}
    for e in examples:
        if seen.setdefault(e.group, e.split) != e.split:
            raise AssertionError(f"group {e.group} appears in {seen[e.group]} and {e.split}")
