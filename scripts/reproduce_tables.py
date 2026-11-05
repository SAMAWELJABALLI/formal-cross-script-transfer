"""Level A: regenerate the paper's LaTeX tables and number macros from archived files and compare.

Runs scripts/paper/make_tables.py and scripts/paper/make_tables_v3.py (outputs under $FORMAL_OUT/tables), then
compares every regenerated table with tables/ byte for byte and every \\newcommand macro with the archived
tables/numbers.tex and tables/v3_numbers.tex.

Without the locally rebuilt corpus (data/examples.jsonl, see docs/DATASETS.md) a few corpus-dependent outputs
cannot be regenerated: tables/data_stats.tex and six count macros (nPairs, nTarcAvail, nTest*Az). They are
reported as "needs corpus", not as failures. tables/compute.tex is a hand-written wrapper around
compute_rows.tex and is not generated.

Usage:  python scripts/reproduce_tables.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT = Path(os.environ.get("FORMAL_OUT", "out"))
HAND_WRITTEN = {"compute.tex"}
CORPUS_TABLES = {"data_stats.tex"}
CORPUS_MACROS = {"nPairs", "nTarcAvail", "nTestmycsentAz", "nTestnarabizisentAz", "nTestnarabizitopicAz",
                 "nTesttsacsentAz"}
MACRO = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$")


def macros(path: Path) -> dict[str, str]:
    return {m.group(1): m.group(2) for line in path.read_text().splitlines() if (m := MACRO.match(line))}


def main():
    env = {**os.environ, "FORMAL_OUT": str(OUT), "MPLBACKEND": "Agg"}
    for script in ("make_tables.py", "make_tables_v3.py"):
        print(f"$ python scripts/paper/{script}", flush=True)
        subprocess.run([sys.executable, f"scripts/paper/{script}"], check=True, env=env)
    have_corpus = Path("data/examples.jsonl").exists()
    ok = True
    print()
    for arch in sorted(Path("tables").glob("*.tex")):
        regen = OUT / "tables" / arch.name
        if arch.name in HAND_WRITTEN:
            print(f"[SKIP]     {arch}  (hand-written wrapper, not generated)")
        elif not regen.exists():
            needs = arch.name in CORPUS_TABLES and not have_corpus
            ok &= needs
            print(f"[{'NEEDS CORPUS' if needs else 'MISSING'}] {arch}")
        elif arch.name.endswith("numbers.tex"):
            a, r = macros(arch), macros(regen)
            same = sum(a[k] == r[k] for k in a if k in r)
            differ = sorted(k for k in a if k in r and a[k] != r[k])
            missing = sorted(set(a) - set(r))
            extra = sorted(set(r) - set(a))
            unexplained = [k for k in missing if not (k in CORPUS_MACROS and not have_corpus)]
            good = not differ and not extra and not unexplained
            ok &= good
            note = f"; {len(missing)} need the corpus ({', '.join(missing)})" if missing and not unexplained else ""
            print(f"[{'OK' if good else 'MISMATCH'}]       {arch}  ({same}/{len(a)} macros identical{note})")
            for k in differ:
                print(f"    {k}: archived {a[k]!r} regenerated {r[k]!r}")
            for k in unexplained + extra:
                print(f"    {k}: {'missing' if k in unexplained else 'unexpected'}")
        else:
            same = regen.read_bytes() == arch.read_bytes()
            ok &= same
            print(f"[{'OK' if same else 'MISMATCH'}]       {arch}  ({'identical bytes' if same else 'content differs'})")
    print("\nALL TABLES REPRODUCED" if ok else "\nSOME TABLES DIFFER (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
