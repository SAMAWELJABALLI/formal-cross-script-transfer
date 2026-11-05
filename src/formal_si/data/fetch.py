"""Download public datasets into data/raw/<name>/ and record provenance.

Each source is a git repository. Nothing here is synthetic. If the network blocks a
host, the failure is recorded in data/raw/fetch_log.json and the run stops with a
clear message; it never falls back to generated data.

Usage: python -m formal_si.data.fetch [--only narabizi tarc ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

SOURCES = {
    # name: (git url, paper)
    "narabizi": ("https://github.com/SamiaTouileb/Narabizi",
                 "Touileb & Barnes, Findings ACL 2021, arXiv:2105.07400"),
    "tarc": ("https://github.com/eligugliotta/tarc",
             "Gugliotta & Dinarelli, LREC 2022, arXiv:2207.04796"),
    "tunizi": ("https://github.com/chaymafourati/TUNIZI-Sentiment-Analysis-Tunisian-Arabizi-Dataset",
               "Fourati et al. 2020, arXiv:2004.14303"),
    "tsac": ("https://github.com/fbougares/TSAC", "Medhaffar et al., WANLP 2017"),
    "myc": ("https://github.com/MouadJb/MYC", "Jbel et al. 2023, arXiv:2303.15987"),
}

# Upstream commits whose build reproduces the archived corpus byte for byte (data/examples.jsonl SHA-256
# 52d9e7b970f3d48fc70d910cd6a2d6f5bd20a50702d757bdad50432deb607b59, checked on 2026-11-05). The original
# fetch cloned each repository's default branch without recording the commit; these were recorded for the
# public release. A clone whose HEAD differs is moved to the pinned commit.
PINNED = {
    "narabizi": "1434f57bbaa718994f9080e4402f72d81a2962b2",
    "tarc": "ee2b293c5af994ff8f03cedbf8d1868f3bcd9689",
    "tunizi": "33a341ce27d7db58198e40469f7cdc229ab3db89",
    "tsac": "3995e702139450fe977841af1700abcdd4c789e7",
    "myc": "645d51af6dce7bae3f27ffc3005e4d788d5ee2e4",
}


def _pin(dest: Path, commit: str) -> str:
    git = ["git", "-C", str(dest)]
    head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head == commit:
        return "pinned"
    subprocess.run(git + ["fetch", "--depth", "1", "origin", commit], capture_output=True, text=True)
    proc = subprocess.run(git + ["checkout", "--detach", commit], capture_output=True, text=True)
    return "pinned (moved from %s)" % head[:12] if proc.returncode == 0 else "PIN FAILED: " + proc.stderr[-200:]


def fetch(names, root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    log = {}
    for name in names:
        url, paper = SOURCES[name]
        dest = root / name
        entry = {"url": url, "paper": paper, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if dest.exists():
            entry["status"] = "present"
        else:
            proc = subprocess.run(["git", "clone", "--depth", "1", url, str(dest)],
                                  capture_output=True, text=True)
            entry["status"] = "ok" if proc.returncode == 0 else "failed"
            entry["stderr"] = proc.stderr[-500:]
        if dest.exists() and name in PINNED:
            entry["pin"] = _pin(dest, PINNED[name])
        if dest.exists():
            head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                                  capture_output=True, text=True)
            entry["commit"] = head.stdout.strip()
            entry["license_files"] = sorted(p.name for p in dest.iterdir()
                                            if p.name.lower().startswith(("license", "copying")))
        log[name] = entry
    (root / "fetch_log.json").write_text(json.dumps(log, indent=2))
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(SOURCES))
    ap.add_argument("--root", default="data/raw")
    args = ap.parse_args()
    log = fetch(args.only, Path(args.root))
    failed = [k for k, v in log.items() if v["status"] == "failed"]
    print(json.dumps({k: v["status"] for k, v in log.items()}, indent=2))
    if failed:
        raise SystemExit(f"Download failed for {failed}; see {args.root}/fetch_log.json. "
                         "No synthetic fallback is used.")


if __name__ == "__main__":
    main()
