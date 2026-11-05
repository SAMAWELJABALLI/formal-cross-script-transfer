"""Confirmatory experiment (protocol/CONFIRMATORY_PROTOCOL.md). CPU only, frozen encoders.

  python -m formal_si.confirm texts            # list strings still missing from the caches
  python -m formal_si.confirm encode --encoder e5-small labse xlmr-base
  python -m formal_si.confirm evaluate --split dev    # all systems on dev
  python -m formal_si.confirm curve                   # learning curve, dev only
  python -m formal_si.confirm evaluate --split test   # THE single test evaluation
  python -m formal_si.confirm stats --split test
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data.audit import load_jsonl
from .eval.metrics import macro_f1
from .methods.projections import difference_scatter
from .text.normalize import deromanize, normalize_arabic
from .translit import cross_fitted, full_dictionary, resources
from .views import apply_emoji

ENCODERS = ["e5-small", "labse", "xlmr-base"]
EMOJI = ["myc", "tsac", "tunizi", "narabizi", "tarc"]
TASKS = [("narabizi_sent", "sentiment", "narabizi"), ("tsac_sent", "sentiment", "tsac"),
         ("myc_sent", "sentiment", "myc"), ("narabizi_topic", "topic", "narabizi")]
EVALS = {"in_script": ["arabic"], "cross_script": ["arabizi"], "codeswitch": ["codeswitch"]}
K, CS, N_RES = 16, [0.1, 1.0, 10.0], 5
OUT = Path("runs/confirm")


def seeds():
    rng = np.random.default_rng(0)
    return [int(rng.integers(1 << 31)) for _ in range(N_RES)]


def load(include_test: bool):
    splits = {"train", "dev", "test"} if include_test else {"train", "dev"}
    return [e for e in apply_emoji(load_jsonl(Path("data/examples.jsonl")), EMOJI) if e.split in splits]


class Views:
    def __init__(self, ex_train_dev):
        self.d = full_dictionary(ex_train_dev)

    def __call__(self, text, form, view):
        if view == "raw":
            return text
        if form == "arabic":
            return normalize_arabic(text)
        return normalize_arabic(deromanize(text)) if view == "rule" else self.d.transliterate(text)


def pair_sets(ex):
    _, nar = resources(ex)
    cf = cross_fitted(ex)
    groups = sorted(nar)
    return groups, {
        "human": {g: (nar[g]["arabic"], nar[g]["arabizi"]) for g in groups},
        "rule": {g: (normalize_arabic(deromanize(nar[g]["arabizi"])), nar[g]["arabizi"]) for g in groups},
        "dict": {g: (cf[g], nar[g]["arabizi"]) for g in groups},
    }


class Store:
    def __init__(self, enc):
        self.parts = []
        for root in ("cache", "cache_confirm"):
            d = Path(root) / enc
            if (d / "emb.npy").exists():
                texts = json.loads((d / "texts.json").read_text())
                self.parts.append((np.load(d / "emb.npy").astype(np.float64), {t: i for i, t in enumerate(texts)}))

    def has(self, t):
        return any(t in idx for _, idx in self.parts)

    def __call__(self, texts):
        out = []
        for t in texts:
            for emb, idx in self.parts:
                if t in idx:
                    out.append(emb[idx[t]])
                    break
            else:
                raise KeyError(t[:60])
        return np.array(out)


def needed_texts(split_eval):
    ex = load(include_test=(split_eval == "test"))
    v = Views([e for e in ex if e.split != "test"])
    texts = set()
    for _, task, src in TASKS:
        for e in ex:
            if e.source == src and e.task == task and e.label and e.split in ("train", split_eval):
                for view in ("raw", "rule", "dict"):
                    texts.add(v(e.text, e.form, view))
    _, ps = pair_sets([e for e in ex if e.split != "test"])
    for s in ps.values():
        for a, b in s.values():
            texts |= {a, b}
    return sorted(t for t in texts if t)


def cmd_texts(args):
    for split in ("dev", "test"):
        t = needed_texts(split)
        st = Store(ENCODERS[0])
        miss = [x for x in t if not st.has(x)]
        print(split, len(t), "needed;", len(miss), "missing from cache")


def cmd_encode(args):
    import torch
    from .encode import ENCODERS as E, encode_texts
    torch.set_num_threads(args.threads)
    texts = sorted(set(needed_texts("test")) | set(needed_texts("dev")))
    for enc in args.encoder:
        st = Store(enc)
        miss = [t for t in texts if not st.has(t)]
        t0 = time.time()
        emb = encode_texts(miss, *E[enc]) if miss else np.zeros((0, 1))
        d = Path("cache_confirm") / enc
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "emb.npy", emb.astype(np.float16))
        (d / "texts.json").write_text(json.dumps(miss, ensure_ascii=False))
        meta = {"encoder": enc, "n_texts": len(miss), "seconds": round(time.time() - t0, 1), "threads": args.threads}
        (d / "meta.json").write_text(json.dumps(meta))
        print(json.dumps(meta), flush=True)


def removal(st, pairs, k):
    a = st([p[0] for p in pairs])
    b = st([p[1] for p in pairs])
    vals, vecs = np.linalg.eigh(difference_scatter(a, b))
    U = vecs[:, np.argsort(vals)[::-1][:min(k, len(pairs))]]
    return lambda x: x - (x @ U) @ U.T


def fit_eval(xtr, ytr, xev, yev, sel_x, sel_y, tf):
    """C chosen on in-script dev (sel_x, sel_y). Returns preds per eval set."""
    htr = tf(xtr)
    best = None
    for c in CS:
        clf = LogisticRegression(C=c, max_iter=3000).fit(htr, ytr)
        f = macro_f1(sel_y, clf.predict(tf(sel_x)))
        if best is None or f > best[0]:
            best = (f, c, clf)
    _, c, clf = best
    return c, {n: clf.predict(tf(x)).tolist() for n, x in xev.items()}


def task_rows(ex, task, src, split):
    rows = [e for e in ex if e.source == src and e.task == task and e.label]
    tr = [e for e in rows if e.split == "train" and e.form == "arabic"]
    ev = {n: [e for e in rows if e.split == split and e.form in f] for n, f in EVALS.items()}
    sel = [e for e in rows if e.split == "dev" and e.form == "arabic"]
    return tr, {n: r for n, r in ev.items() if r}, sel


def cmd_evaluate(args):
    split = args.split
    ex = load(include_test=(split == "test"))
    trdev = [e for e in ex if e.split != "test"]
    v = Views(trdev)
    groups, ps = pair_sets(trdev)
    out = OUT / split
    out.mkdir(parents=True, exist_ok=True)
    pred_f = open(out / "predictions.jsonl", "w")
    gold = {}
    sd = seeds()
    t_start = time.time()
    for tname, task, src in TASKS:
        tr, ev, sel = task_rows(ex, task, src, split)
        gold[tname] = {n: {"labels": [str(e.label) for e in r], "groups": [e.group for e in r]} for n, r in ev.items()}
        ytr_full = np.array([e.label for e in tr])
        sel_y = [e.label for e in sel]
        subs = [np.random.default_rng(s).choice(len(tr), int(0.9 * len(tr)), replace=False) for s in sd]
        # Char n-gram systems (no encoder)
        for view, name in (("rule", "S5_char_rule"), ("dict", "S5b_char_dict"), ("raw", "S6_char_raw")):
            for r, sub in enumerate(subs):
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True)
                x = vec.fit_transform([v(tr[i].text, tr[i].form, view) for i in sub])
                clf = LogisticRegression(C=1.0, max_iter=3000).fit(x, ytr_full[sub])
                preds = {n: clf.predict(vec.transform([v(e.text, e.form, view) for e in rows])).tolist()
                         for n, rows in ev.items()}
                pred_f.write(json.dumps({"task": tname, "encoder": "none", "system": name, "resample": r,
                                         "C": 1.0, "preds": preds}) + "\n")
        for enc in ENCODERS:
            st = Store(enc)
            tfs = {"S0_raw": ("raw", None), "S1_human": ("raw", "human"), "S2_rule_pairs": ("raw", "rule"),
                   "S3_dict_pairs": ("raw", "dict"), "S4_derom_rule": ("rule", None), "S4b_derom_dict": ("dict", None)}
            for name, (view, pairs) in tfs.items():
                xtr = st([v(e.text, e.form, view) for e in tr])
                xev = {n: st([v(e.text, e.form, view) for e in rows]) for n, rows in ev.items()}
                sx = st([v(e.text, e.form, view) for e in sel])
                tf = removal(st, list(ps[pairs].values()), K) if pairs else (lambda x: x)
                for r, sub in enumerate(subs):
                    c, preds = fit_eval(xtr[sub], ytr_full[sub], xev, None, sx, sel_y, tf)
                    pred_f.write(json.dumps({"task": tname, "encoder": enc, "system": name, "resample": r,
                                             "C": c, "preds": preds}) + "\n")
            pred_f.flush()
            print(tname, enc, "done", round(time.time() - t_start), "s", flush=True)
    pred_f.close()
    (out / "gold.json").write_text(json.dumps(gold))
    (out / "timing.json").write_text(json.dumps({"seconds": round(time.time() - t_start, 1)}))


def cmd_curve(args):
    ex = load(include_test=False)
    groups, ps = pair_sets(ex)
    out = OUT / "curve_dev"
    out.mkdir(parents=True, exist_ok=True)
    rows_out = open(out / "metrics.jsonl", "w")
    sd = seeds()
    for tname, task, src in TASKS:
        tr, ev, sel = task_rows(ex, task, src, "dev")
        ytr = np.array([e.label for e in tr])
        sel_y = [e.label for e in sel]
        subs = [np.random.default_rng(s).choice(len(tr), int(0.9 * len(tr)), replace=False) for s in sd]
        for enc in ENCODERS:
            st = Store(enc)
            xtr = st([e.text for e in tr])
            xev = {"cross_script": st([e.text for e in ev["cross_script"]]),
                   "in_script": st([e.text for e in ev["in_script"]])}
            yev = {n: [e.label for e in ev[n]] for n in xev}
            sx = st([e.text for e in sel])
            for n in (50, 100, 250, 500, 998):
                for draw in range(3 if n < 998 else 1):
                    gsel = sorted(np.random.default_rng(1000 + draw).choice(len(groups), n, replace=False))
                    for src_name in ("human", "rule", "dict"):
                        tf = removal(st, [ps[src_name][groups[i]] for i in gsel], K)
                        f = defaultdict(list)
                        for sub in subs:
                            _, preds = fit_eval(xtr[sub], ytr[sub], xev, None, sx, sel_y, tf)
                            for nm in preds:
                                f[nm].append(macro_f1(yev[nm], preds[nm]))
                        rows_out.write(json.dumps({"task": tname, "encoder": enc, "n_pairs": n, "draw": draw,
                                                   "pairs": src_name, **{f"f1_{k}": float(np.mean(v)) for k, v in f.items()}}) + "\n")
                rows_out.flush()
            print(tname, enc, "curve done", flush=True)
    rows_out.close()


def fast_macro_f1(y, p, labels):
    f = []
    for c in labels:
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def boot_draws(gold, preds_a, preds_b, n_boot, seed=0):
    """Grouped paired bootstrap (whole test sentence groups); delta of mean-over-resamples macro-F1.
    Returns summary (x100) and the draws, so per-cell draws can be averaged for the aggregate CI."""
    y = np.array(gold["labels"]); g = np.array(gold["groups"]); labels = sorted(set(y))
    uniq, inv = np.unique(g, return_inverse=True)
    by_g = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    pa = [np.array(p) for p in preds_a]; pb = [np.array(p) for p in preds_b]

    def d(idx):
        return np.mean([fast_macro_f1(y[idx], a[idx], labels) for a in pa]) - \
            np.mean([fast_macro_f1(y[idx], b[idx], labels) for b in pb])

    obs = d(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    ds = np.array([d(np.concatenate([by_g[j] for j in rng.integers(0, len(uniq), len(uniq))]))
                   for _ in range(n_boot)])
    lo, hi = np.quantile(ds, [0.025, 0.975])
    p = min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()))
    return {"delta": 100 * obs, "lo": 100 * lo, "hi": 100 * hi, "p": float(p)}, 100 * ds


def cmd_stats(args):
    from .eval.stats import holm
    d = OUT / args.split
    gold = json.loads((d / "gold.json").read_text())
    P = defaultdict(list)
    for l in open(d / "predictions.jsonl"):
        r = json.loads(l)
        P[(r["task"], r["encoder"], r["system"])].append(r["preds"])
    cells = [(t, e) for t, _, _ in TASKS for e in ENCODERS]
    summ = {}
    for (t, e, s), pl in P.items():
        for n in gold[t]:
            y = gold[t][n]["labels"]
            summ.setdefault(f"{t}|{e}|{s}", {})[n] = 100 * float(np.mean([macro_f1(y, p[n]) for p in pl]))
    hyps = {"H1 S1>S0": ("S1_human", "S0_raw"), "H2 S1>S2": ("S1_human", "S2_rule_pairs"),
            "H3 S1>S3": ("S1_human", "S3_dict_pairs"), "sec S1 vs S4": ("S1_human", "S4_derom_rule"),
            "sec S1 vs S4b": ("S1_human", "S4b_derom_dict"), "sec S4b vs S4": ("S4b_derom_dict", "S4_derom_rule"),
            "sec S3 vs S2": ("S3_dict_pairs", "S2_rule_pairs")}
    res = {}
    for h, (a, b) in hyps.items():
        per, draws = {}, []
        for t, e in cells:
            g = gold[t]["cross_script"]
            r, dr = boot_draws(g, [p["cross_script"] for p in P[(t, e, a)]],
                               [p["cross_script"] for p in P[(t, e, b)]], args.n_boot, seed=0)
            per[f"{t}|{e}"] = r
            draws.append(dr)
        adj = holm({k: v["p"] for k, v in per.items()})
        agg = np.mean(draws, axis=0)
        res[h] = {"cells": {k: {**v, "p_holm": adj[k]["p_holm"]} for k, v in per.items()},
                  "mean_delta": float(np.mean([v["delta"] for v in per.values()])),
                  "agg_lo": float(np.quantile(agg, 0.025)), "agg_hi": float(np.quantile(agg, 0.975)),
                  "n_cells_ci_above_0": sum(v["lo"] > 0 for v in per.values()),
                  "n_cells_ci_below_0": sum(v["hi"] < 0 for v in per.values()),
                  "n_cells_holm_sig": sum(adj[k]["p_holm"] < 0.05 for k in per)}
    (d / "stats.json").write_text(json.dumps({"summary": summ, "hypotheses": res}, indent=1, default=float))
    for h, v in res.items():
        print(f"{h}: mean {v['mean_delta']:+.2f} [{v['agg_lo']:+.2f}, {v['agg_hi']:+.2f}]  cells>0 {v['n_cells_ci_above_0']} <0 {v['n_cells_ci_below_0']} holm {v['n_cells_holm_sig']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("texts")
    e = sub.add_parser("encode")
    e.add_argument("--encoder", nargs="+", default=ENCODERS)
    e.add_argument("--threads", type=int, default=4)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--split", choices=["dev", "test"], required=True)
    sub.add_parser("curve")
    s = sub.add_parser("stats")
    s.add_argument("--split", choices=["dev", "test"], required=True)
    s.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    {"texts": cmd_texts, "encode": cmd_encode, "evaluate": cmd_evaluate, "curve": cmd_curve, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    main()
