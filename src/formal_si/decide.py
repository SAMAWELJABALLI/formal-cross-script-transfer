"""Apply the pre-registered go/no-go gates (Section 7) to dev metrics.

Usage: python -m formal_si.decide --runs runs/pilot --out reports/decision.md
Thresholds are fixed constants here so they cannot drift after results are seen.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

G1_MIN_GAP = 5.0          # macro-F1 points, raw in-script minus cross-script
G1_MIN_ENCODERS = 2
G2_MIN_AUC_LOSS = 0.05
G2_MIN_DIALECT_LOSS = 3.0
G3_MIN_GAIN = 2.0
G3_MAX_AUC_LOSS = 0.02
UNCONDITIONAL = ("leace", "lsar", "centering", "splince", "adapter")
# Input-side canonicalisation baseline (Section 8): G1 is judged against it, and FORMAL-SI
# must beat it as well as the unconditional erasers in G3. "Closed" reuses G1_MIN_GAP.
INPUT_SIDE = ("derom_encode",)


def summarise(path: Path) -> dict:
    """Mean over resamples per (method, setting); keeps per-resample values."""
    groups = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        key = (r["method"], tuple(sorted((k, r[k]) for k in ("k", "lam0", "beta", "r", "rank")
                                         if k in r)))
        groups[key].append(r)
    out = {}
    for key, rs in groups.items():
        metrics = {m for r in rs for m in r if isinstance(r[m], float)}
        out[key] = {m: [r[m] for r in rs if m in r] for m in metrics}
    return out


def mean(v):
    return float(np.mean(v)) if v else float("nan")


def decide(run_dir: Path, in_name="in_script", cross_name="cross_script") -> tuple[str, dict]:
    lines, facts, gaps = [], {}, []
    for enc_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        s = summarise(enc_dir / "metrics.jsonl")
        raw = next(v for k, v in s.items() if k[0] == "raw")
        gap = 100 * (mean(raw[f"f1_{in_name}"]) - mean(raw[f"f1_{cross_name}"]))
        f = facts[enc_dir.name] = {"raw_gap_points": round(gap, 2)}
        derom = next((v for k, v in s.items() if k[0] in INPUT_SIDE), None)
        if derom is not None:
            # Residual gap left after deromanize-then-encode, against the raw in-script F1.
            dgap = 100 * (mean(raw[f"f1_{in_name}"]) - mean(derom[f"f1_{cross_name}"]))
            f["derom_residual_gap_points"] = round(dgap, 2)
            gap = min(gap, dgap)
        gaps.append(gap)
        raw_auc, raw_dia = mean(raw.get("negation_auc", [])), mean(raw.get("dialect_f1", []))
        best_unc, best_unc_f1 = None, -1
        tradeoff = False
        for k, v in s.items():
            if k[0] in UNCONDITIONAL + INPUT_SIDE:
                f1 = mean(v[f"f1_{cross_name}"])
                if f1 > best_unc_f1:
                    best_unc, best_unc_f1 = k, f1
                if k[0] in INPUT_SIDE:
                    continue  # G2 is about erasure methods only; derom_encode counts for G3
                closes = f1 > mean(raw[f"f1_{cross_name}"])
                auc_loss = raw_auc - mean(v.get("negation_auc", []))
                dia_loss = 100 * (raw_dia - mean(v.get("dialect_f1", [])))
                if closes and (auc_loss >= G2_MIN_AUC_LOSS or dia_loss >= G2_MIN_DIALECT_LOSS):
                    tradeoff = True
        f["g2_tradeoff"] = tradeoff
        si = {k: v for k, v in s.items() if k[0] == "formal_si"}
        if si and best_unc:
            k_best = max(si, key=lambda k: mean(si[k][f"f1_{cross_name}"]))
            gain = 100 * (mean(si[k_best][f"f1_{cross_name}"]) - best_unc_f1)
            auc_loss = raw_auc - mean(si[k_best].get("negation_auc", []))
            f.update(best_formal_si=str(k_best), best_unconditional=str(best_unc),
                     gain_points=round(gain, 2), formal_si_auc_loss=round(auc_loss, 4))
            f["g3_pass_point_estimate"] = bool(gain >= G3_MIN_GAIN and
                                               (np.isnan(auc_loss) or auc_loss <= G3_MAX_AUC_LOSS))
    g1 = sum(g >= G1_MIN_GAP for g in gaps) >= G1_MIN_ENCODERS
    g2 = any(v.get("g2_tradeoff") for v in facts.values())
    g3 = any(v.get("g3_pass_point_estimate") for v in facts.values())
    lines.append(f"G1 problem exists: {'PASS' if g1 else 'FAIL'} (gap left after the better of "
                 f"raw and deromanize-then-encode: {[round(g, 2) for g in gaps]})")
    lines.append(f"G2 trade-off exists: {'PASS' if g2 else 'FAIL'}")
    lines.append(f"G3 method helps (point estimate; confirm with grouped bootstrap): "
                 f"{'PASS' if g3 else 'FAIL'}")
    verdict = "GO" if (g1 and g2 and g3) else "NO-GO"
    lines.append(f"Verdict on dev data: {verdict}")
    return "\n".join(lines), facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/pilot")
    ap.add_argument("--out", default="reports/decision.md")
    args = ap.parse_args()
    text, facts = decide(Path(args.runs))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n\n```json\n" + json.dumps(facts, indent=2) + "\n```\n")
    print(text)


if __name__ == "__main__":
    main()
