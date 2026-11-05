"""Per-corpus analysis of the frozen-encoder pilot (dev split only).

For every (corpus run, encoder) it reports each method family under two selection rules:
  * oracle: the hyperparameters with the best mean cross-script dev F1 in that cell
    (what decide.py uses; optimistic, since selection and evaluation share dev data);
  * loco:   leave-one-corpus-out -- the hyperparameters with the best mean cross-script
    dev F1 averaged over the OTHER per-corpus runs, same encoder (no selection on the
    cell being evaluated).
Then a grouped paired bootstrap (whole sentence groups resampled, delta averaged over
the training resamples inside each draw) compares FORMAL-SI with the strongest baseline,
with Holm correction across cells, and isolates where the gain comes from:
  invariance-only (lam0 = 0) vs selective (lam0 > 0), random removal at matched k,
  and FORMAL-SI on top of deromanization.

Usage: python -m formal_si.analyze --runs runs/pilot_narabizi runs/pilot_tsac ... --out reports/pilot_analysis.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .eval.metrics import macro_f1
from .eval.stats import holm

HP = ("k", "lam0", "beta", "r")
BASELINES = ("raw", "derom_encode", "centering", "lsar", "leace", "splince", "random")
METRICS = ("f1_in_script", "f1_cross_script", "f1_codeswitch", "negation_auc",
           "negation_flip_rate", "invariance", "dialect_f1", "fit_seconds")


def family(method, hp):
    if method == "formal_si":
        if hp.get("lam0", 0) == 0:
            return "formal_si_inv_only"          # beta has no effect when lam0 = 0
        return "formal_si_sel_contrast" if hp.get("beta", 0) > 0 else "formal_si_sel_task"
    if method == "derom+formal_si":
        return "derom+formal_si"
    return method


def load(enc_dir: Path):
    rows = [json.loads(l) for l in open(enc_dir / "metrics.jsonl")]
    preds = [json.loads(l) for l in open(enc_dir / "dev_predictions.jsonl")]
    gold = json.loads((enc_dir / "dev_gold.json").read_text())
    cfgs = defaultdict(lambda: {"rows": [], "preds": []})
    for r, p in zip(rows, preds):
        hp = tuple((k, r[k]) for k in HP if k in r)
        key = (r["method"], hp)
        cfgs[key]["rows"].append(r)
        cfgs[key]["preds"].append(p["preds"])
    return cfgs, gold


def mean_metric(c, m):
    v = [r[m] for r in c["rows"] if m in r and r[m] == r[m]]
    return float(np.mean(v)) if v else float("nan")


def boot_delta(gold, preds_a, preds_b, n_boot=2000, seed=0):
    """Grouped paired bootstrap of mean-over-resamples macro-F1(a) - macro-F1(b)."""
    y = np.array(gold["labels"])
    g = np.array(gold["groups"])
    uniq, inv = np.unique(g, return_inverse=True)
    by_g = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    pa = [np.array(p) for p in preds_a]
    pb = [np.array(p) for p in preds_b]

    def d(idx):
        return np.mean([macro_f1(y[idx], a[idx]) for a in pa]) - \
            np.mean([macro_f1(y[idx], b[idx]) for b in pb])

    obs = d(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([by_g[j] for j in rng.integers(0, len(uniq), len(uniq))])
        ds[i] = d(idx)
    lo, hi = np.quantile(ds, [0.025, 0.975])
    p = min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()))
    return {"delta": 100 * obs, "lo": 100 * lo, "hi": 100 * hi, "p": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="per-corpus run dirs (LOCO pool)")
    ap.add_argument("--extra-runs", nargs="*", default=[], help="e.g. pooled run: oracle only")
    ap.add_argument("--out", default="reports/pilot_analysis.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    data = {}  # (run, enc) -> (cfgs, gold)
    for rd in [*args.runs, *args.extra_runs]:
        for enc_dir in sorted(p for p in Path(rd).iterdir() if (p / "metrics.jsonl").exists()):
            data[(Path(rd).name, enc_dir.name)] = load(enc_dir)
    per_corpus = {Path(r).name for r in args.runs}

    def pick(cell, fam, rule):
        cfgs, _ = data[cell]
        cands = [k for k in cfgs if family(k[0], dict(k[1])) == fam]
        if not cands:
            return None
        if rule == "loco" and cell[0] in per_corpus:
            others = [c for c in data if c[1] == cell[1] and c[0] in per_corpus and c != cell]
            # only the "sentiment" runs share a label space; topic is still a valid pool for
            # choosing hyperparameters since only the mean F1 ranking is used
            def score(k):
                v = [mean_metric(data[o][0][k], "f1_cross_script") for o in others if k in data[o][0]]
                return np.mean(v) if v else -1
        else:
            def score(k):
                return mean_metric(cfgs[k], "f1_cross_script")
        return max(cands, key=score)

    fams = [*BASELINES, "formal_si_inv_only", "formal_si_sel_task", "formal_si_sel_contrast",
            "derom+leace", "derom+formal_si"]
    lines = ["# FORMAL pilot: per-corpus analysis (dev split only)", "",
             "Macro-F1 x 100, mean over 5 training resamples. Cross = train Arabic script, "
             "evaluate human Arabizi. AUC = negation sensitivity (share of anchors whose rule "
             "negation is farther than their human other-script paraphrase; higher = more "
             "sensitive). Inv = invariance score on dev human pairs.", ""]
    tests, summary = {}, {}
    for rule in ("loco", "oracle"):
        lines += [f"## Selection rule: {rule}", ""]
        for cell in sorted(data):
            if rule == "loco" and cell[0] not in per_corpus:
                continue
            cfgs, gold = data[cell]
            lines += [f"### {cell[0]} / {cell[1]}", "",
                      "| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |",
                      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            chosen = {}
            for fam in fams:
                k = pick(cell, fam, rule)
                if k is None:
                    continue
                chosen[fam] = k
                c = cfgs[k]
                f = lambda m, s=100: (f"{s * mean_metric(c, m):.1f}" if mean_metric(c, m) == mean_metric(c, m) else "")
                lines.append(f"| {fam} | {dict(k[1]) or '-'} | {f('f1_in_script')} | "
                             f"{f('f1_cross_script')} | {f('f1_codeswitch')} | "
                             f"{mean_metric(c, 'negation_auc'):.3f} | "
                             f"{mean_metric(c, 'invariance'):.3f} | {f('dialect_f1')} | "
                             f"{mean_metric(c, 'fit_seconds'):.2f} |")
            lines.append("")
            g = gold.get("cross_script")
            if not g:
                continue
            base_fams = [b for b in BASELINES if b in chosen]
            best_base = max(base_fams, key=lambda b: mean_metric(cfgs[chosen[b]], "f1_cross_script"))
            comps = {}
            si_all = [f for f in ("formal_si_inv_only", "formal_si_sel_task", "formal_si_sel_contrast") if f in chosen]
            best_si = max(si_all, key=lambda f: mean_metric(cfgs[chosen[f]], "f1_cross_script"))
            P = lambda fam: [p["cross_script"] for p in cfgs[chosen[fam]]["preds"]]
            comps[f"{best_si} vs best baseline ({best_base})"] = (best_si, best_base)
            comps["selective (best lam0>0) vs invariance-only"] = (
                max([f for f in si_all if f != "formal_si_inv_only"],
                    key=lambda f: mean_metric(cfgs[chosen[f]], "f1_cross_script")), "formal_si_inv_only")
            comps["invariance-only vs derom_encode"] = ("formal_si_inv_only", "derom_encode")
            if "derom+formal_si" in chosen:
                comps["derom+formal_si vs derom_encode"] = ("derom+formal_si", "derom_encode")
            lines += ["| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |",
                      "| --- | ---: | --- | ---: | ---: |"]
            for name, (a, b) in comps.items():
                r = boot_delta(g, P(a), P(b), args.n_boot)
                dauc = mean_metric(cfgs[chosen[a]], "negation_auc") - mean_metric(cfgs[chosen[b]], "negation_auc")
                tests[f"{rule}|{cell[0]}|{cell[1]}|{name}"] = r
                lines.append(f"| {name} | {r['delta']:+.1f} | [{r['lo']:+.1f}, {r['hi']:+.1f}] | "
                             f"{r['p']:.3f} | {dauc:+.3f} |")
            # Random removal at the same k as the chosen invariance-only setting.
            k_inv = dict(chosen["formal_si_inv_only"][1]).get("k")
            rnd = [k for k in cfgs if k[0] == "random" and dict(k[1]).get("k") == k_inv]
            if rnd:
                r = boot_delta(g, P("formal_si_inv_only"),
                               [p["cross_script"] for p in cfgs[rnd[0]]["preds"]], args.n_boot)
                tests[f"{rule}|{cell[0]}|{cell[1]}|inv-only vs random k={k_inv}"] = r
                lines.append(f"| invariance-only vs random removal (k={k_inv}) | {r['delta']:+.1f} | "
                             f"[{r['lo']:+.1f}, {r['hi']:+.1f}] | {r['p']:.3f} | |")
            lines.append("")
            summary[f"{rule}|{cell[0]}|{cell[1]}"] = {
                fam: {"setting": dict(chosen[fam][1]),
                      **{m: mean_metric(cfgs[chosen[fam]], m) for m in METRICS}} for fam in chosen}
    # Holm within each selection rule, over the "vs best baseline" tests.
    for rule in ("loco", "oracle"):
        fam_tests = {k: v["p"] for k, v in tests.items() if k.startswith(rule) and "best baseline" in k}
        if fam_tests:
            adj = holm(fam_tests)
            lines += [f"## Holm-corrected: FORMAL-SI vs best baseline ({rule})", "",
                      "| cell | delta | p | p_holm | reject |", "| --- | ---: | ---: | ---: | --- |"]
            for k, v in adj.items():
                lines.append(f"| {k.split('|', 1)[1]} | {tests[k]['delta']:+.1f} | {tests[k]['p']:.3f} | "
                             f"{v.get('p_adj', v.get('p_holm', float('nan'))):.3f} | {v.get('reject')} |")
            lines.append("")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"tests": tests, "summary": summary}, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
