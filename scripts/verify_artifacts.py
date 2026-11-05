"""Verify the integrity and provenance records of the archived artifacts. Pure standard library; no fitting.

Checks
  1. artifacts/MANIFEST.sha256: every listed file exists and has the listed SHA-256.
  2. protocols/confirmatory/CONFIRMATORY_PROTOCOL.sha256 and protocols/exploratory/POST_CONFIRMATORY_PROTOCOL_V2.sha256
     match the protocol texts.
  3. protocols/confirmatory/code_at_test_run.sha256 (code hashes recorded at the confirmatory test run) match
     src/formal_si/confirm.py and src/formal_si/translit.py.
  4. Archived prediction files carry predicted labels and short metadata only (no free text).

Usage:  python scripts/verify_artifacts.py            # verify
        python scripts/verify_artifacts.py --write    # (maintainers) rewrite artifacts/MANIFEST.sha256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
MANIFEST = Path("artifacts/MANIFEST.sha256")
COVERED = ["protocols", "runs", "reports", "artifacts", "tables", "figures", "configs", "src", "scripts", "tests"]
SKIP_PARTS = {"__pycache__", ".pytest_cache"}
PRED_KEYS = {"task", "encoder", "system", "resample", "C", "preds"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def covered_files() -> list[Path]:
    files = []
    for top in COVERED:
        for p in sorted(Path(top).rglob("*")):
            if p.is_file() and p != MANIFEST and not SKIP_PARTS & set(p.parts) and p.suffix != ".pyc":
                files.append(p)
    return files


def write_manifest():
    lines = [f"{sha256(p)}  {p.as_posix()}" for p in covered_files()]
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"wrote {MANIFEST} ({len(lines)} files)")


def check_manifest() -> bool:
    ok, n, bad = True, 0, []
    listed = set()
    for line in MANIFEST.read_text().splitlines():
        digest, path = line.split("  ", 1)
        listed.add(path)
        p = Path(path)
        n += 1
        if not p.is_file():
            bad.append(f"missing: {path}")
        elif sha256(p) != digest:
            bad.append(f"hash mismatch: {path}")
    unlisted = [p.as_posix() for p in covered_files() if p.as_posix() not in listed]
    for b in bad:
        print("   ", b)
    for u in unlisted:
        print("    not in manifest:", u)
    ok = not bad and not unlisted
    print(f"[{'OK' if ok else 'FAIL'}] manifest: {n} files listed, {len(bad)} problems, {len(unlisted)} unlisted")
    return ok


def check_record(record: Path, base: Path) -> bool:
    """A sha256sum-style record (optionally followed by a timestamp line); paths resolved against base."""
    ok = True
    for line in record.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            continue  # timestamp line
        digest, rel = parts
        p = base / rel
        if not p.is_file():  # records written in the original layout (e.g. protocol/<file>): match by file name
            p = base / Path(rel).name
        good = p.is_file() and sha256(p) == digest
        ok &= good
        print(f"[{'OK' if good else 'FAIL'}] {record.name}: {p.as_posix()} {digest[:12]}...")
    return ok


def check_predictions() -> bool:
    """Every archived prediction file holds predicted label strings and short metadata only, no text."""
    ok, n, files, values = True, 0, 0, set()
    for p in sorted(Path("runs").rglob("*predictions*.jsonl")):
        files += 1
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                n += 1
                if p.parts[1] in ("confirm", "confirm2") and not set(r) <= PRED_KEYS:
                    ok = False
                    print(f"[FAIL] unexpected keys {sorted(set(r) - PRED_KEYS)} in {p}")
                    break
                for preds in r.get("preds", {}).values():
                    values.update(preds)
                long = [k for k, v in r.items() if isinstance(v, str) and len(v) > 64]
                if long:
                    ok = False
                    print(f"[FAIL] long string field(s) {long} in {p}")
                    break
    labels_only = len(values) <= 20 and all(isinstance(v, str) and len(v) <= 12 for v in values)
    ok &= labels_only
    print(f"[{'OK' if ok else 'FAIL'}] prediction files: {files} files, {n} records; predicted values {sorted(values)}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="rewrite the manifest instead of checking it")
    a = ap.parse_args()
    if a.write:
        write_manifest()
        return
    ok = check_manifest()
    ok &= check_record(Path("protocols/confirmatory/CONFIRMATORY_PROTOCOL.sha256"), Path("protocols/confirmatory"))
    ok &= check_record(Path("protocols/exploratory/POST_CONFIRMATORY_PROTOCOL_V2.sha256"), Path("protocols/exploratory"))
    ok &= check_record(Path("protocols/confirmatory/code_at_test_run.sha256"), Path("src"))
    ok &= check_predictions()
    print("ALL ARTIFACT CHECKS PASSED" if ok else "SOME ARTIFACT CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
