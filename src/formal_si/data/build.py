"""Parse the downloaded corpora into data/examples.jsonl (one Example per row).

Formats were confirmed by inspecting the cloned repositories (commits recorded in
data/raw/fetch_log.json):

- NArabizi (Touileb & Barnes 2021): data/Narabizi/pos/{train,dev,test}_NArabizi.conllu,
  12 columns: ID, Arabizi form, French gloss, Arabic-script form, code-switched form,
  UPOS, ... ; '# text =' holds the original Arabizi sentence. Sentiment / topic labels in
  data/Narabizi/{sentiment,topic}/<split>_Narabizi_<task>.txt as '# sent_id = X<TAB>LABEL'.
- TArC (Gugliotta & Dinarelli 2022): tarc.tsv, one token per line, '<eos>' rows end a
  sentence; columns arabish and CODA (CODA keeps 'foreign' tokens in Latin script, so
  the CODA sentence is a code-switched Arabic-script form). No task labels.
- TSAC (Medhaffar et al. 2017): {train,test}_{pos,neg}.txt, one comment per line.
- TUNIZI (Fourati et al. 2020): 'label;text' lines, label 1 / -1.
- MYC (Jbel et al. 2023): DATA_CLEANED.csv, UTF-16, columns sentence, polarity (1 / -1).

Usage: python -m formal_si.data.build --raw data/raw --out data/examples.jsonl
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path

from ..text.normalize import detect_script
from .schema import Example, check_no_leakage, grouped_split, merge_groups

MALFORMED: list[tuple[str, str]] = []
FORM_OF_SCRIPT = {"arab": "arabic", "latn": "arabizi", "mixed": "mixed", "none": "mixed"}


def _narabizi_conllu(path: Path) -> dict[str, dict[str, str]]:
    sents, cur, sid = {}, None, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# sent_id"):
            if cur is not None and sid is not None:  # sentence not closed by a blank line
                sents[sid] = {"arabizi": cur["arabizi"], "arabic": " ".join(cur["arabic"]),
                              "codeswitch": " ".join(cur["codeswitch"])}
            sid = line.split("=", 1)[1].strip()
            cur = {"arabizi": None, "arabic": [], "codeswitch": []}
        elif line.startswith("# text") and cur is not None:
            cur["arabizi"] = line.split("=", 1)[1].strip()
        elif line and not line.startswith("#") and cur is not None:
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:
                continue
            if len(cols) < 5:
                # One stray line ('fif') in train_NArabizi.conllu at commit 1434f57;
                # recorded in the audit rather than guessed at.
                MALFORMED.append((path.name, line))
                continue
            for key, col in (("arabic", 3), ("codeswitch", 4)):
                if cols[col] not in ("_", ""):
                    cur[key].append(cols[col])
        elif not line and cur is not None and sid is not None:
            sents[sid] = {"arabizi": cur["arabizi"], "arabic": " ".join(cur["arabic"]),
                          "codeswitch": " ".join(cur["codeswitch"])}
            cur, sid = None, None
    if cur is not None and sid is not None:
        sents[sid] = {"arabizi": cur["arabizi"], "arabic": " ".join(cur["arabic"]),
                      "codeswitch": " ".join(cur["codeswitch"])}
    return sents


def _labels(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if "\t" not in line:
            continue
        sid, lab = line.split("\t", 1)
        out[sid.split("=", 1)[1].strip()] = lab.strip()
    return out


def load_narabizi(root: Path) -> list[Example]:
    base = root / "narabizi" / "data" / "Narabizi"
    out = []
    for split in ("train", "dev", "test"):
        sents = _narabizi_conllu(base / "pos" / f"{split}_NArabizi.conllu")
        labels = {t: _labels(base / t / f"{split}_Narabizi_{t}.txt") for t in ("sentiment", "topic")}
        for sid, forms in sents.items():
            for task in ("sentiment", "topic"):
                lab = labels[task].get(sid)
                if lab is None:
                    continue
                if task == "sentiment":
                    lab = {"POS": "pos", "NEG": "neg"}.get(lab)  # NEU / MIX dropped
                    if lab is None:
                        continue
                if task == "topic" and lab.upper() == "PRAYER":
                    lab = "RELIGION"  # merged as in Touileb & Barnes (2021)
                for form, text in forms.items():
                    if text:
                        out.append(Example(uid=f"narabizi:{task}:{form}:{sid}", text=text,
                                           group=f"narabizi:{sid}", form=form, dialect="DZ",
                                           source="narabizi", task=task, label=lab, split=split))
    return out


def load_tarc(root: Path) -> list[Example]:
    """Invariance pairs only (task 'pairs', no label)."""
    rows = (root / "tarc" / "tarc.tsv").read_text(encoding="utf-8").splitlines()
    header = rows[0].split("\t")
    ia, ic = header.index("arabish"), header.index("CODA")
    out, a, c, n = [], [], [], 0
    for line in rows[1:]:
        cols = line.split("\t")
        if len(cols) <= max(ia, ic):
            continue
        if cols[ia] == "<eos>":
            if a:
                g = f"tarc:{n}"
                out.append(Example(uid=f"{g}:arabizi", text=" ".join(a), group=g, form="arabizi",
                                   dialect="TN", source="tarc", task="pairs", label=None))
                out.append(Example(uid=f"{g}:codeswitch", text=" ".join(c), group=g,
                                   form="codeswitch", dialect="TN", source="tarc",
                                   task="pairs", label=None))
                n += 1
            a, c = [], []
            continue
        a.append(cols[ia])
        c.append(cols[ic])
    return out


def _single(source, dialect, items):
    out = []
    for i, (text, label, split) in enumerate(items):
        text = text.strip()
        if not text:
            continue
        uid = f"{source}:{i}"
        out.append(Example(uid=uid, text=text, group=uid,
                           form=FORM_OF_SCRIPT[detect_script(text)], dialect=dialect,
                           source=source, task="sentiment", label=label, split=split))
    return out


def load_tsac(root: Path) -> list[Example]:
    items = []
    for split in ("train", "test"):
        for pol in ("pos", "neg"):
            for line in (root / "tsac" / f"{split}_{pol}.txt").read_text(encoding="utf-8").splitlines():
                items.append((line, pol, "test" if split == "test" else None))
    return _single("tsac", "TN", items)


def load_tunizi(root: Path) -> list[Example]:
    items, bad = [], 0
    for line in (root / "tunizi" / "TUNIZI-Dataset.txt").read_text(encoding="utf-8").splitlines():
        lab, _, text = line.partition(";")
        lab = {"1": "pos", "-1": "neg"}.get(lab.strip())
        if lab is None:
            bad += 1
            continue
        items.append((text, lab, None))
    return _single("tunizi", "TN", items)


def load_myc(root: Path) -> list[Example]:
    raw = (root / "myc" / "DATA_CLEANED.csv").read_bytes().decode("utf-16")
    rows = list(csv.reader(io.StringIO(raw)))
    items = []
    for r in rows[1:]:
        if len(r) < 2:
            continue
        lab = {"1": "pos", "-1": "neg"}.get(r[-1].strip())
        if lab:
            items.append((",".join(r[:-1]), lab, None))
    return _single("myc", "MA", items)


def build(raw: Path) -> list[Example]:
    exs = (load_narabizi(raw) + load_tarc(raw) + load_tsac(raw) + load_tunizi(raw)
           + load_myc(raw))
    # Near-duplicate grouping within each single-form corpus only: NArabizi/TArC groups
    # are already defined by sentence id, and merging across corpora would join unrelated
    # short comments ("hhhh") from different sources.
    by_src: dict[str, list[Example]] = {}
    for e in exs:
        by_src.setdefault(e.source, []).append(e)
    for src in ("tsac", "tunizi", "myc"):
        merge_groups(by_src[src], threshold=0.8)
        for e in by_src[src]:
            e.group = f"{src}:{e.group}"
    for src, rows in by_src.items():
        grouped_split(rows, dev=0.1, test=0.1 if src != "tsac" else 0.0, seed=0)
    check_no_leakage(exs)
    return exs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/examples.jsonl")
    args = ap.parse_args()
    exs = build(Path(args.raw))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for e in exs:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
    for k, v in sorted(Counter((e.source, e.task, e.form, e.split) for e in exs).items()):
        print(*k, v, sep="\t")
    print("malformed lines skipped:", MALFORMED)


if __name__ == "__main__":
    main()
