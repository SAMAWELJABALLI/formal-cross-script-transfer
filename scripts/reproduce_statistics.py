"""Level A: recompute every statistical output from the archived predictions and compare with the archive.

No model is fitted and no encoder is loaded. Each step runs the original analysis code (src/formal_si) on a
copy of the archived run directory under $FORMAL_OUT/stats_work, so the archived files are never written.

Steps
  confirmatory     formal_si.confirm stats       runs/confirm/{dev,test}/stats.json   (frozen protocol analysis)
  depstats         formal_si.depstats            reports/depstats/test.json           (post-hoc sensitivity analysis)
  post-conf        formal_si.confirm2 stats      runs/confirm2/{dev,test}/stats.json  (post-confirmatory, exploratory)
  v3-predictions   scripts/paper/v3_prediction_analysis.py  reports/v3/predictions_analysis.json (descriptive)
  pilot            formal_si.analyze             reports/pilot/pilot_analysis.json    (pre-registration pilot, dev only)
  hc               formal_si.hc_analyze          reports/hc/hc_analysis.json          (pre-registration H-C, dev only)

Usage:  python scripts/reproduce_statistics.py [--steps confirmatory depstats ...] [--n-boot N]
Exit status 0 only if every regenerated file equals its archived counterpart (exact JSON equality, or
numerical equality within --atol when --atol > 0).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT = Path(os.environ.get("FORMAL_OUT", "out"))
WORK = OUT / "stats_work"
STEPS = ["confirmatory", "depstats", "post-conf", "v3-predictions", "pilot", "hc"]
DEFAULT_STEPS = ["confirmatory", "depstats", "post-conf", "v3-predictions", "hc"]
# Bootstrap draws behind each archived file. The protocol fixes 2,000 draws for the test analysis; the archived
# confirmatory *dev* check (runs/confirm/dev/stats.json) was computed with 1,000 draws, which this script
# reproduces byte for byte. formal_si.hc_analyze uses its own default of 1,000.
ARCHIVED_N_BOOT = {("confirmatory", "dev"): 1000}


def compare(a, b, atol, path="$"):
    """List of differences between two JSON values (exact unless atol > 0)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}: keys differ {sorted(set(a) ^ set(b))[:5]}"] if set(a) != set(b) else []
        for k in set(a) & set(b):
            out += compare(a[k], b[k], atol, f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in compare(x, y, atol, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        same = a == b or (a != a and b != b) or (atol > 0 and abs(a - b) <= atol)
        return [] if same else [f"{path}: {a!r} != {b!r}"]
    return [] if a == b else [f"{path}: {str(a)[:60]!r} != {str(b)[:60]!r}"]


def check(regenerated: Path, archived: Path, atol: float) -> tuple[bool, str]:
    a, b = json.loads(regenerated.read_text()), json.loads(archived.read_text())
    diffs = compare(a, b, atol)
    exact = regenerated.read_bytes() == archived.read_bytes()
    if not diffs:
        return True, "identical bytes" if exact else "identical values"
    return False, f"{len(diffs)} differences, e.g. {diffs[:3]}"


def stage(src: Path, dst: Path, patterns=("gold.json", "predictions*.jsonl")):
    dst.mkdir(parents=True, exist_ok=True)
    for pat in patterns:
        for f in src.glob(pat):
            shutil.copy2(f, dst / f.name)


def quiet(fn, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args)
    return buf.getvalue()


def run_step(step: str, n_boot: int) -> list[tuple[Path, Path]]:
    """Run one step; return (regenerated, archived) pairs to compare."""
    from formal_si import confirm, confirm2, depstats
    pairs = []
    if step == "confirmatory":
        confirm.OUT = WORK / "confirm"
        for split in ("dev", "test"):
            stage(Path(f"runs/confirm/{split}"), confirm.OUT / split)
            nb = n_boot or ARCHIVED_N_BOOT.get(("confirmatory", split), 2000)
            quiet(confirm.cmd_stats, argparse.Namespace(split=split, n_boot=nb))
            pairs.append((confirm.OUT / split / "stats.json", Path(f"runs/confirm/{split}/stats.json")))
    elif step == "depstats":
        d = WORK / "depstats"
        stage(Path("runs/confirm/test"), d)
        res = depstats.run(d, depstats.HYPS, n_boot or 2000)
        (d / "test.json").write_text(json.dumps(res, indent=1))
        pairs.append((d / "test.json", Path("reports/depstats/test.json")))
    elif step == "post-conf":
        confirm2.OUT = WORK / "confirm2"
        for split in ("dev", "test"):
            stage(Path(f"runs/confirm2/{split}"), confirm2.OUT / split)
            quiet(confirm2.cmd_stats, argparse.Namespace(split=split, n_boot=n_boot or 2000))
            pairs.append((confirm2.OUT / split / "stats.json", Path(f"runs/confirm2/{split}/stats.json")))
    elif step == "v3-predictions":
        subprocess.run([sys.executable, "scripts/paper/v3_prediction_analysis.py"], check=True, stdout=subprocess.DEVNULL)
        pairs.append((OUT / "reports/v3/predictions_analysis.json", Path("reports/v3/predictions_analysis.json")))
    elif step == "pilot":
        out = OUT / "reports/pilot/pilot_analysis.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "formal_si.analyze", "--runs", "runs/pilot_narabizi", "runs/pilot_tsac",
                        "runs/pilot_myc", "runs/pilot_narabizi_topic", "--extra-runs", "runs/pilot",
                        "--out", str(out), "--n-boot", str(n_boot or 2000)], check=True, stdout=subprocess.DEVNULL)
        pairs.append((out.with_suffix(".json"), Path("reports/pilot/pilot_analysis.json")))
    elif step == "hc":
        out = OUT / "reports/hc/hc_analysis.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "formal_si.hc_analyze", "--runs", "runs/hc",
                        "--pilot", "reports/pilot/pilot_analysis.json", "--out", str(out),
                        "--n-boot", str(n_boot or 1000)],
                       check=True, stdout=subprocess.DEVNULL)
        pairs.append((out.with_suffix(".json"), Path("reports/hc/hc_analysis.json")))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=DEFAULT_STEPS,
                    help="default: every step except 'pilot', which takes well over an hour on 4 cores")
    ap.add_argument("--n-boot", type=int, default=None,
                    help="override bootstrap draws for every step (default: the draws used for each archived file)")
    ap.add_argument("--atol", type=float, default=0.0, help="absolute tolerance for numbers (default: exact)")
    a = ap.parse_args()
    ok_all = True
    for step in a.steps:
        t0 = time.time()
        for regen, arch in run_step(step, a.n_boot):
            ok, msg = check(regen, arch, a.atol)
            ok_all &= ok
            print(f"[{'OK' if ok else 'MISMATCH'}] {step:15s} {arch}  ({msg}; {time.time() - t0:.0f} s)", flush=True)
    print("ALL STATISTICS REPRODUCED" if ok_all else "SOME STATISTICS DIFFER (see above)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
