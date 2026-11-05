"""Dependency-aware re-analysis of saved confirmatory predictions (no refitting, no new test access).

The 12 confirmatory cells are not independent: the three encoders share the same test sentences, and the
93 NArabizi sentiment test groups are a subset of the 144 NArabizi topic test groups. This module therefore
uses a *joint* grouped bootstrap: in each draw, sentence groups are resampled once per corpus
(NArabizi, TSAC, MYC) and the same resample is applied to every task and encoder of that corpus.
It reports, from the same draws:
  * the 12-cell mean (identical estimand to the frozen protocol, CI now from joint draws);
  * task-level effects (mean over encoders) with Holm over 4 tasks;
  * corpus-level effects (NArabizi = mean of its two tasks) and the equally weighted corpus mean;
  * leave-one-task-out and leave-one-encoder-out means;
  * a cell-weighted vs corpus-weighted comparison.
This is a post-hoc sensitivity analysis of the frozen confirmatory analysis, not a replacement for it.

Usage: python -m formal_si.depstats --run runs/confirm/test --out reports/depstats/test.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .confirm import fast_macro_f1
from .eval.stats import holm

CORPUS = {"narabizi_sent": "narabizi", "narabizi_topic": "narabizi", "tsac_sent": "tsac", "myc_sent": "myc"}
TASKS = ["narabizi_sent", "tsac_sent", "myc_sent", "narabizi_topic"]
HYPS = {"H1 S1>S0": ("S1_human", "S0_raw"), "H2 S1>S2": ("S1_human", "S2_rule_pairs"),
        "H3 S1>S3": ("S1_human", "S3_dict_pairs"), "sec S1 vs S4": ("S1_human", "S4_derom_rule"),
        "sec S1 vs S4b": ("S1_human", "S4b_derom_dict")}


def q(x):
    lo, hi = np.quantile(x, [0.025, 0.975])
    p = min(1.0, 2 * min((x <= 0).mean(), (x >= 0).mean()))
    return {"mean": float(np.mean(x)), "lo": float(lo), "hi": float(hi), "p": float(p)}


def run(run_dir, hyps, n_boot=2000, seed=0, encoders=None, eval_name="cross_script"):
    d = Path(run_dir)
    gold = json.loads((d / "gold.json").read_text())
    P = defaultdict(list)
    for l in open(d / "predictions.jsonl"):
        r = json.loads(l)
        P[(r["task"], r["encoder"], r["system"])].append(np.array(r["preds"][eval_name]))
    encs = encoders or sorted({e for (_, e, _) in P if e != "none"})
    # corpus-level group index
    cg = {}
    for t in TASKS:
        cg.setdefault(CORPUS[t], set()).update(gold[t][eval_name]["groups"])
    cg = {c: sorted(s) for c, s in cg.items()}
    rows_of = {}
    for t in TASKS:
        g = np.array(gold[t][eval_name]["groups"])
        pos = {k: i for i, k in enumerate(cg[CORPUS[t]])}
        gi = np.array([pos[x] for x in g])
        rows_of[t] = [np.flatnonzero(gi == i) for i in range(len(cg[CORPUS[t]]))]
    rng = np.random.default_rng(seed)
    draws_idx = []
    for _ in range(n_boot):
        pick = {c: rng.integers(0, len(v), len(v)) for c, v in cg.items()}
        draws_idx.append({t: np.concatenate([rows_of[t][j] for j in pick[CORPUS[t]]]) for t in TASKS})
    out = {}
    for h, (a, b) in hyps.items():
        cells = [(t, e) for t in TASKS for e in encs if (t, e, a) in P and (t, e, b) in P]
        obs, D = {}, {}
        for t, e in cells:
            y = np.array(gold[t][eval_name]["labels"])
            labs = sorted(set(y))

            def delta(idx):
                return 100 * (np.mean([fast_macro_f1(y[idx], p[idx], labs) for p in P[(t, e, a)]])
                              - np.mean([fast_macro_f1(y[idx], p[idx], labs) for p in P[(t, e, b)]]))
            obs[(t, e)] = delta(np.arange(len(y)))
            D[(t, e)] = np.array([delta(di[t]) for di in draws_idx])
        cell_mean = np.mean([D[c] for c in cells], axis=0)
        task = {t: np.mean([D[(t, e)] for e in encs if (t, e) in D], axis=0) for t in TASKS}
        task_obs = {t: float(np.mean([obs[(t, e)] for e in encs if (t, e) in obs])) for t in TASKS}
        corp = {"narabizi": (task["narabizi_sent"] + task["narabizi_topic"]) / 2, "tsac": task["tsac_sent"],
                "myc": task["myc_sent"]}
        corp_obs = {"narabizi": (task_obs["narabizi_sent"] + task_obs["narabizi_topic"]) / 2,
                    "tsac": task_obs["tsac_sent"], "myc": task_obs["myc_sent"]}
        task_res = {t: {**q(task[t]), "observed": task_obs[t]} for t in TASKS}
        adj = holm({t: v["p"] for t, v in task_res.items()})
        for t in TASKS:
            task_res[t]["p_holm_4tasks"] = adj[t]["p_holm"]
        r = {"n_cells": len(cells),
             "cell_mean_joint": {**q(cell_mean), "observed": float(np.mean(list(obs.values())))},
             "tasks": task_res,
             "corpora": {c: {**q(v), "observed": corp_obs[c]} for c, v in corp.items()},
             "corpus_weighted_mean": {**q(np.mean(list(corp.values()), axis=0)),
                                      "observed": float(np.mean(list(corp_obs.values())))},
             "out_of_narabizi_mean": {**q((corp["tsac"] + corp["myc"]) / 2),
                                      "observed": (corp_obs["tsac"] + corp_obs["myc"]) / 2},
             "leave_one_task_out": {t: {**q(np.mean([task[u] for u in TASKS if u != t], axis=0)),
                                        "observed": float(np.mean([task_obs[u] for u in TASKS if u != t]))}
                                    for t in TASKS},
             "leave_one_encoder_out": {}}
        for e in encs:
            keep = [c for c in cells if c[1] != e]
            r["leave_one_encoder_out"][e] = {**q(np.mean([D[c] for c in keep], axis=0)),
                                             "observed": float(np.mean([obs[c] for c in keep]))}
        r["n_tasks_ci_above_0"] = sum(v["lo"] > 0 for v in task_res.values())
        r["n_tasks_holm_sig"] = sum(v["p_holm_4tasks"] < 0.05 for v in task_res.values())
        out[h] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/confirm/test")
    ap.add_argument("--out", default="reports/depstats/test.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    res = run(a.run, HYPS, a.n_boot)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for h, r in res.items():
        c, w = r["cell_mean_joint"], r["corpus_weighted_mean"]
        print(f"{h}: cells {c['observed']:+.2f} [{c['lo']:+.2f},{c['hi']:+.2f}] | corpus-wt {w['observed']:+.2f} "
              f"[{w['lo']:+.2f},{w['hi']:+.2f}] | tasks CI>0 {r['n_tasks_ci_above_0']}/4 holm {r['n_tasks_holm_sig']} | "
              f"out-of-NArabizi {r['out_of_narabizi_mean']['observed']:+.2f} [{r['out_of_narabizi_mean']['lo']:+.2f},"
              f"{r['out_of_narabizi_mean']['hi']:+.2f}]")


if __name__ == "__main__":
    main()
