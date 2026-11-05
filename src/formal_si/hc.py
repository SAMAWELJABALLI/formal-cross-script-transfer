"""H-C: when does cross-script invariance help or hurt? (dev split only, cached embeddings)

Invariance-only removal (FORMAL-SI with lam0 = 0, i.e. remove the top-k eigenvectors of the
pair-difference scatter S_inv) with S_inv estimated from different pair sources:

  human_ar_az   NArabizi train groups: human Arabic-script vs human Arabizi
  synth_ar_az   the SAME Arabizi sentences paired with their rule deromanization
                (identical Latin side; only the Arabic side differs: human vs synthetic)
  human_az_cs   NArabizi + TArC train groups: human Arabizi vs code-switched layer
  human_all     every pair of distinct forms in NArabizi + TArC train groups (pilot setting)

All sources are subsampled to the same number of pairs (the smallest), so differences are
not a count effect. Per cell it also logs task/script entanglement
E_k = tr(U_k^T S_task U_k) / tr(S_task), S_task = between-class scatter of the Arabic-script
training rows, U_k = top-k eigenvectors of S_inv(human_all).

Usage: python -m formal_si.hc --encoder e5-small --out runs/hc
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

from .data.audit import load_jsonl
from .eval import metrics as M
from .methods.projections import FormalSI, between_class_scatter, difference_scatter
from .run import EmbeddingStore
from .views import apply_emoji, view_text

CORPORA = ["pilot_narabizi", "pilot_tsac", "pilot_myc", "pilot_narabizi_topic"]
KS = [4, 16, 64]
CS = [0.1, 1.0, 10.0]


def group_forms(rows):
    g = defaultdict(dict)
    for e in rows:
        g[e.group].setdefault(e.form, e.text)
    return g


def pair_sources(examples, rng, n_match=None):
    tr = [e for e in examples if e.split == "train" and e.source in ("narabizi", "tarc")]
    g = group_forms(tr)
    src = {"human_ar_az": [], "synth_ar_az": [], "human_az_cs": [], "human_all": []}
    for fm in g.values():
        if "arabic" in fm and "arabizi" in fm:
            src["human_ar_az"].append((fm["arabic"], fm["arabizi"]))
            src["synth_ar_az"].append((view_text(fm["arabizi"], "arabizi", "deromanized"),
                                       fm["arabizi"]))
        if "arabizi" in fm and "codeswitch" in fm:
            src["human_az_cs"].append((fm["arabizi"], fm["codeswitch"]))
        for f1, f2 in itertools.combinations(sorted(fm), 2):
            src["human_all"].append((fm[f1], fm[f2]))
    n = n_match or min(len(v) for v in src.values())
    out = {}
    for k, v in src.items():
        idx = rng.choice(len(v), n, replace=False)
        out[k] = [v[i] for i in sorted(idx)]
    return out, {k: len(v) for k, v in src.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="runs/hc")
    ap.add_argument("--resamples", type=int, default=5)
    args = ap.parse_args()
    st = EmbeddingStore(Path(args.cache) / args.encoder)
    base = yaml.safe_load(open(Path(args.configs) / "pilot.yaml"))
    ex = [e for e in apply_emoji(load_jsonl(Path(base["examples"])), base["strip_emoji_sources"])
          if e.split in ("train", "dev")]
    rng = np.random.default_rng(0)
    pairs, sizes, n = pair_sources(ex, rng)
    scat = {k: difference_scatter(st([a for a, _ in v]), st([b for _, b in v]))
            for k, v in pairs.items()}
    U = {}
    for k, s in scat.items():
        vals, vecs = np.linalg.eigh(s)
        U[k] = vecs[:, np.argsort(vals)[::-1]]
    for cname in CORPORA:
        cfg = yaml.safe_load(open(Path(args.configs) / f"{cname}.yaml"))
        rows = [e for e in ex if e.source in cfg["sources"] and e.task == cfg["task"] and e.label]
        train = [e for e in rows if e.split == "train" and e.form in cfg["train_forms"]]
        evals = {nm: [e for e in rows if e.split == "dev" and e.form in f]
                 for nm, f in cfg["eval_sets"].items()}
        x_full, y_full = st([e.text for e in train]), np.array([e.label for e in train])
        x_ev = {nm: st([e.text for e in r]) for nm, r in evals.items() if r}
        y_ev = {nm: [e.label for e in r] for nm, r in evals.items() if r}
        od = Path(args.out) / args.encoder / cname
        od.mkdir(parents=True, exist_ok=True)
        (od / "dev_gold.json").write_text(json.dumps(
            {nm: {"labels": [str(e.label) for e in r], "groups": [e.group for e in r]}
             for nm, r in evals.items() if r}))
        s_task = between_class_scatter(x_full, y_full)
        ent = {f"E_{k}_{src}": float(np.trace(U[src][:, :k].T @ s_task @ U[src][:, :k]) / np.trace(s_task))
               for k in KS for src in U}
        (od / "entanglement.json").write_text(json.dumps(
            {**ent, "pair_counts_available": sizes, "pairs_used_each": n,
             "random_expected": {k: k / x_full.shape[1] for k in KS}}, indent=1))
        log, plog = open(od / "metrics.jsonl", "w"), open(od / "dev_predictions.jsonl", "w")
        rr = np.random.default_rng(cfg.get("seed", 0))
        seeds = [int(rr.integers(1 << 31)) for _ in range(args.resamples)]
        for r, seed in enumerate(seeds):
            sub = np.random.default_rng(seed).choice(len(train), int(0.9 * len(train)), replace=False)
            x_tr, y_tr = x_full[sub], y_full[sub]
            for src in ["none", *pairs]:
                for k in (KS if src != "none" else [0]):
                    t0 = time.time()
                    if src == "none":
                        tf = lambda x: x
                    else:
                        m = FormalSI(k=k, lam0=0.0, beta=0.0)
                        m.fit(x_tr, y_tr, st([p[0] for p in pairs[src]]), st([p[1] for p in pairs[src]]))
                        tf = m.transform
                    h_tr = tf(x_tr)
                    best = None
                    for c in CS:
                        clf = LogisticRegression(C=c, max_iter=2000).fit(h_tr, y_tr)
                        pr = {nm: clf.predict(tf(x)) for nm, x in x_ev.items()}
                        sc = {f"f1_{nm}": M.macro_f1(y_ev[nm], p) for nm, p in pr.items()}
                        if best is None or sc["f1_in_script"] > best[0]["f1_in_script"]:
                            best = (sc, c, pr)
                    sc, c, pr = best
                    res = {"pairs": src, "k": k, "resample": r, "seed": seed, "C": c, **sc,
                           "seconds": round(time.time() - t0, 2)}
                    log.write(json.dumps(res) + "\n")
                    plog.write(json.dumps({"pairs": src, "k": k, "resample": r,
                                           "preds": {nm: [str(v) for v in p] for nm, p in pr.items()}}) + "\n")
            log.flush()
            plog.flush()
        log.close()
        plog.close()
        print(cname, "done", flush=True)


if __name__ == "__main__":
    main()
