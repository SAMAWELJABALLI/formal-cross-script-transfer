"""Readers that turn downloaded files into ``Example`` records.

The column names and file paths for each dataset live in configs/datasets.yaml, not
in code, because they must be confirmed against the actual downloaded files (they
could not be inspected from the environment where this package was written).
Readers fail loudly on missing columns rather than guessing.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..text.normalize import detect_script
from .schema import Example


def read_table(path: Path, *, text_col: str, label_col: str | None, delimiter: str = ",",
               encoding: str = "utf-8", has_header: bool = True,
               fieldnames: list[str] | None = None) -> list[dict]:
    with open(path, encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter,
                                fieldnames=None if has_header else fieldnames)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no rows")
    missing = [c for c in (text_col, label_col) if c and c not in rows[0]]
    if missing:
        raise KeyError(f"{path}: columns {missing} not found; have {list(rows[0])}")
    return rows


def table_examples(name: str, spec: dict, root: Path) -> list[Example]:
    """spec keys: files ({split: relative path}), text_col, label_col, dialect, task,
    optional delimiter/has_header/fieldnames/label_map/form (else detected per row)."""
    out: list[Example] = []
    for split, rel in spec["files"].items():
        rows = read_table(root / name / rel, text_col=spec["text_col"],
                          label_col=spec.get("label_col"),
                          delimiter=spec.get("delimiter", ","),
                          has_header=spec.get("has_header", True),
                          fieldnames=spec.get("fieldnames"))
        for i, r in enumerate(rows):
            text = (r[spec["text_col"]] or "").strip()
            if not text:
                continue
            label = r.get(spec.get("label_col") or "", None)
            if spec.get("label_map"):
                label = spec["label_map"].get(str(label).strip(), None)
                if label is None:
                    continue  # e.g. neutral / mixed dropped for binary sentiment
            script = detect_script(text)
            form = spec.get("form") or {"arab": "arabic", "latn": "arabizi"}.get(script, "mixed")
            uid = f"{name}:{split}:{i}"
            out.append(Example(uid=uid, text=text, group=uid, form=form,
                               dialect=spec["dialect"], source=name, task=spec["task"],
                               label=None if label is None else str(label),
                               split=None if split == "all" else split))
    return out


def read_conllu_sentences(path: Path, form_field: str = "FORM",
                          misc_key: str | None = None) -> list[dict]:
    """Minimal CoNLL-U reader returning [{'id': sent_id, 'tokens': [...], 'meta': {...}}].
    With ``misc_key``, the token is taken from that key in the MISC column (used for
    parallel transliteration layers); otherwise from FORM."""
    cols = ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"]
    sents, toks, meta = [], [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if toks:
                    sents.append({"id": meta.get("sent_id", str(len(sents))),
                                  "tokens": toks, "meta": meta})
                toks, meta = [], {}
                continue
            if line.startswith("#"):
                if "=" in line:
                    k, v = line[1:].split("=", 1)
                    meta[k.strip()] = v.strip()
                continue
            parts = line.split("\t")
            if "-" in parts[0] or "." in parts[0]:
                continue
            row = dict(zip(cols, parts))
            if misc_key:
                misc = dict(kv.split("=", 1) for kv in row.get("MISC", "").split("|") if "=" in kv)
                tok = misc.get(misc_key)
                if tok is None:
                    raise KeyError(f"{path}: MISC key {misc_key!r} missing in line {line[:80]!r}")
                toks.append(tok)
            else:
                toks.append(row[form_field])
    if toks:
        sents.append({"id": meta.get("sent_id", str(len(sents))), "tokens": toks, "meta": meta})
    return sents


def parallel_examples(name: str, layers: dict[str, list[dict]], labels: dict[str, str],
                      *, dialect: str, task: str, splits: dict[str, str]) -> list[Example]:
    """Build one group per sentence id from aligned layers {form: [sentences]}.
    Only sentence ids present in every layer are kept."""
    ids = set.intersection(*[{s["id"] for s in v} for v in layers.values()])
    out = []
    for form, sents in layers.items():
        for s in sents:
            if s["id"] not in ids:
                continue
            out.append(Example(uid=f"{name}:{form}:{s['id']}", text=" ".join(s["tokens"]),
                               group=f"{name}:{s['id']}", form=form, dialect=dialect,
                               source=name, task=task, label=labels.get(s["id"]),
                               split=splits.get(s["id"])))
    return out
