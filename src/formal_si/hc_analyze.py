"""Analysis for H-C (dev only). Primary k = 16, fixed before the H-C results were seen
(the k most often chosen by leave-one-corpus-out selection in the pilot).

Usage: python -m formal_si.hc_analyze --runs runs/hc --pilot reports/pilot/pilot_analysis.json --out reports/hc/hc_analysis.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .analyze import boot_delta
from .eval.stats import holm

PRIMARY_K = 16
SOURCES = ["human_ar_az", "synth_ar_az", "human_az_cs", "human_all"]
COMPARISONS = [("human_ar_az", "none", "human Ar-Az pairs vs no removal"),
               ("synth_ar_az", "human_ar_az", "synthetic vs human Ar-Az pairs"),
               ("human_az_cs", "human_ar_az", "Az-codeswitch vs Ar-Az pairs"),
               ("human_all", "none", "all human pairs vs no removal")]


def load_cell(d: Path):
    rows = [json.loads(l) for l in open(d / "metrics.jsonl")]
    preds = [json.loads(l) for l in open(d / "dev_predictions.jsonl")]
    agg, pr = defaultdict(list), defaultdict(list)
    for r, p in zip(rows, preds):
        agg[(r["pairs"], r["k"])].append(r)
        pr[(r["pairs"], r["k"])].append(p["preds"])
    gold = json.loads((d / "dev_gold.json").read_text())
    ent = json.loads((d / "entanglement.json").read_text())
    return agg, pr, gold, ent


def m(rs, key):
    return 100 * float(np.mean([r[key] for r in rs if key in r]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/hc")
    ap.add_argument("--pilot", default="reports/pilot/pilot_analysis.json")
    ap.add_argument("--out", default="reports/hc/hc_analysis.md")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    pilot = json.load(open(args.pilot))["summary"]
    cells = {}
    for enc_dir in sorted(Path(args.runs).iterdir()):
        for c in sorted(enc_dir.iterdir()):
            if (c / "metrics.jsonl").exists():
                cells[(c.name, enc_dir.name)] = load_cell(c)
    L = ["# H-C: when does cross-script invariance help or hurt? (dev split only)", "",
         f"Invariance-only removal (top-k eigenvectors of the pair-difference scatter), 998 pairs per "
         f"source, 5 resamples, C chosen on in-script dev. Primary k = {PRIMARY_K}. Macro-F1 x 100.", ""]
    L += ["## Cross-script F1 by pair source (k = 16)", "",
          "| corpus | encoder | none | human Ar-Az | synth Ar-Az | human Az-CS | human all | in-script none | in-script human all |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for (c, e), (agg, _, _, _) in cells.items():
        L.append(f"| {c} | {e} | {m(agg[('none', 0)], 'f1_cross_script'):.1f} | " +
                 " | ".join(f"{m(agg[(s, PRIMARY_K)], 'f1_cross_script'):.1f}" for s in SOURCES) +
                 f" | {m(agg[('none', 0)], 'f1_in_script'):.1f} | {m(agg[('human_all', PRIMARY_K)], 'f1_in_script'):.1f} |")
    L.append("")
    tests = {}
    L += ["## Paired comparisons at k = 16 (grouped bootstrap, Holm within each comparison over 12 cells)", ""]
    for a, b, name in COMPARISONS:
        res = {}
        for (c, e), (agg, pr, gold, _) in cells.items():
            kb = 0 if b == "none" else PRIMARY_K
            res[f"{c}|{e}"] = boot_delta(gold["cross_script"],
                                         [p["cross_script"] for p in pr[(a, PRIMARY_K)]],
                                         [p["cross_script"] for p in pr[(b, kb)]], args.n_boot)
        adj = holm({k: v["p"] for k, v in res.items()})
        L += [f"### {name}", "", "| cell | delta | 95% CI | p | p_holm |", "| --- | ---: | --- | ---: | ---: |"]
        for k, v in res.items():
            L.append(f"| {k} | {v['delta']:+.1f} | [{v['lo']:+.1f}, {v['hi']:+.1f}] | {v['p']:.3f} | {adj[k]['p_holm']:.3f} |")
        n_pos = sum(v["lo"] > 0 for v in res.values())
        n_neg = sum(v["hi"] < 0 for v in res.values())
        L += ["", f"Intervals above 0: {n_pos}/12; below 0: {n_neg}/12; mean delta {np.mean([v['delta'] for v in res.values()]):+.2f}.", ""]
        tests[name] = {k: {**v, "p_holm": adj[k]["p_holm"]} for k, v in res.items()}
    # Entanglement / dose-response
    pts = []
    for (c, e), (agg, _, _, ent) in cells.items():
        base = agg[("none", 0)]
        derom = pilot.get(f"loco|{c}|{e}", {}).get("derom_encode", {}).get("f1_cross_script")
        for k in (4, 16, 64):
            rs = agg[("human_all", k)]
            pts.append({"cell": f"{c}|{e}", "encoder": e, "corpus": c, "k": k,
                        "E": ent[f"E_{k}_human_all"],
                        "raw_gap": m(base, "f1_in_script") - m(base, "f1_cross_script"),
                        "d_cross": m(rs, "f1_cross_script") - m(base, "f1_cross_script"),
                        "d_in": m(rs, "f1_in_script") - m(base, "f1_in_script"),
                        "d_vs_derom": (m(rs, "f1_cross_script") - 100 * derom) if derom is not None else None})
    L += ["## Entanglement and dose-response (human all pairs)", "",
          "E_k = share of the Arabic-script training task scatter inside the top-k invariance subspace "
          "(random expectation k/d: 0.010-0.17).", "",
          "| cell | k | E_k | raw gap | d cross | d in-script | cross minus derom-encode |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for p in pts:
        dd = f"{p['d_vs_derom']:+.1f}" if p["d_vs_derom"] is not None else ""
        L.append(f"| {p['cell']} | {p['k']} | {p['E']:.2f} | {p['raw_gap']:.1f} | {p['d_cross']:+.1f} | {p['d_in']:+.1f} | {dd} |")
    corr = {}
    for k in (4, 16, 64, "all"):
        sel = [p for p in pts if k == "all" or p["k"] == k]
        for x, y in [("E", "d_in"), ("E", "d_cross"), ("raw_gap", "d_cross"), ("E", "d_vs_derom")]:
            xs = [p[x] for p in sel if p[y] is not None]
            ys = [p[y] for p in sel if p[y] is not None]
            rho, pv = spearmanr(xs, ys)
            corr[f"k={k}|{x}~{y}"] = {"rho": float(rho), "p": float(pv), "n": len(xs)}
    L += ["", "| k | relation | Spearman rho | p | n |", "| --- | --- | ---: | ---: | ---: |"]
    for key, v in corr.items():
        k, rel = key.split("|")
        L.append(f"| {k} | {rel} | {v['rho']:+.2f} | {v['p']:.3f} | {v['n']} |")
    L.append("")
    L.append("Cells share encoders and corpora, so the correlations are descriptive (n = 12 per k, not independent).")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps({"tests": tests, "points": pts, "corr": corr}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
