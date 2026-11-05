"""POST-CONFIRMATORY exploratory controls (protocol/POST_CONFIRMATORY_PROTOCOL_V2.md, SHA-256 1ca10244...).

Everything here was specified after the v1 test evaluation and is reported as exploratory.
Fixed from v1: tasks, splits, classifier, C selection on in-script dev, seeds, k=16, 998 pairs, bootstrap.

  python -m formal_si.confirm2 encode --encoder e5-small labse xlmr-base   # strings missing from caches
  python -m formal_si.confirm2 evaluate --split dev
  python -m formal_si.confirm2 stats --split dev
  python -m formal_si.confirm2 evaluate --split test   # run once
  python -m formal_si.confirm2 stats --split test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import confirm as C
from .geometry import tarc_pairs
from .text.normalize import normalize_arabic

OUT = Path("runs/confirm2")
NEURAL = Path("cache_neural_translit")
ROOTS = ("cache", "cache_confirm", "cache_v2", "cache_labse_st")


class Store(C.Store):
    def __init__(self, enc):
        self.parts = []
        for root in ROOTS:
            d = Path(root) / enc
            if (d / "emb.npy").exists() and (d / "texts.json").exists():
                texts = json.loads((d / "texts.json").read_text())
                if texts:
                    self.parts.append((np.load(d / "emb.npy").astype(np.float64), {t: i for i, t in enumerate(texts)}))


class Views(C.Views):
    def __init__(self, ex_train_dev):
        super().__init__(ex_train_dev)
        self.nmap = json.loads((NEURAL / "full_input_map.json").read_text())

    def __call__(self, text, form, view):
        if view == "neural":
            return normalize_arabic(text) if form == "arabic" else self.nmap[text]
        return super().__call__(text, form, view)


def pair_sets2(trdev):
    groups, ps = C.pair_sets(trdev)
    _, nar = C.resources(trdev)
    cf = json.loads((NEURAL / "cross_fitted_pairs.json").read_text())
    ps["neural"] = {g: (cf[g], nar[g]["arabizi"]) for g in groups}
    tp = tarc_pairs(trdev, "train", n=len(groups), seed=0)
    ps["tarc"] = {i: p for i, p in enumerate(tp)}
    return groups, ps


# (view, pair source) per system
BASE = {"S0_raw": ("raw", None), "S1_human": ("raw", "human"), "S3_dict_pairs": ("raw", "dict"),
        "S4b_derom_dict": ("dict", None), "S1T_tarc_pairs": ("raw", "tarc"), "S3c_neural_pairs": ("raw", "neural"),
        "S4c_derom_neural": ("neural", None)}
V1_EXTRA = {"S2_rule_pairs": ("raw", "rule"), "S4_derom_rule": ("rule", None)}
SYSTEMS = {"e5-small": BASE, "labse": BASE, "xlmr-base": BASE, "labse-st": {**BASE, **V1_EXTRA}}


def needed(split_eval):
    ex = C.load(include_test=(split_eval == "test"))
    trdev = [e for e in ex if e.split != "test"]
    v = Views(trdev)
    _, ps = pair_sets2(trdev)
    texts = set()
    for _, task, src in C.TASKS:
        for e in ex:
            if e.source == src and e.task == task and e.label and e.split in ("train", "dev", split_eval):
                for view in ("raw", "rule", "dict", "neural"):
                    texts.add(v(e.text, e.form, view))
    for s in ps.values():
        for a, b in s.values():
            texts |= {a, b}
    return sorted(t for t in texts if t)


def cmd_encode(args):
    import torch
    from .encode import ENCODERS as E, encode_texts
    torch.set_num_threads(args.threads)
    texts = sorted(set(needed("test")) | set(needed("dev")))
    for enc in args.encoder:
        st = Store(enc)
        miss = [t for t in texts if not st.has(t)]
        t0 = time.time()
        d = Path("cache_v2") / enc
        d.mkdir(parents=True, exist_ok=True)
        if enc == "labse-st":
            from .encode_labse_st import encode as enc_st
            emb = enc_st(miss, args.threads) if miss else np.zeros((0, 768))
        else:
            emb = encode_texts(miss, *E[enc]) if miss else np.zeros((0, 1))
        np.save(d / "emb.npy", emb.astype(np.float16))
        (d / "texts.json").write_text(json.dumps(miss, ensure_ascii=False))
        meta = {"encoder": enc, "n_texts": len(miss), "seconds": round(time.time() - t0, 1), "threads": args.threads}
        (d / "meta.json").write_text(json.dumps(meta))
        print(json.dumps(meta), flush=True)


def cmd_evaluate(args):
    split = args.split
    ex = C.load(include_test=(split == "test"))
    trdev = [e for e in ex if e.split != "test"]
    v = Views(trdev)
    _, ps = pair_sets2(trdev)
    out = OUT / split
    out.mkdir(parents=True, exist_ok=True)
    pred_f = open(out / f"predictions.{args.part}.jsonl", "w")
    gold = {}
    sd = C.seeds()
    t_start = time.time()
    for tname, task, src in C.TASKS:
        tr, ev, sel = C.task_rows(ex, task, src, split)
        gold[tname] = {n: {"labels": [str(e.label) for e in r], "groups": [e.group for e in r]} for n, r in ev.items()}
        ytr_full = np.array([e.label for e in tr])
        sel_y = [e.label for e in sel]
        subs = [np.random.default_rng(s).choice(len(tr), int(0.9 * len(tr)), replace=False) for s in sd]
        for r, sub in enumerate(subs if args.char else []):
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True)
            x = vec.fit_transform([v(tr[i].text, tr[i].form, "neural") for i in sub])
            clf = LogisticRegression(C=1.0, max_iter=3000).fit(x, ytr_full[sub])
            preds = {n: clf.predict(vec.transform([v(e.text, e.form, "neural") for e in rows])).tolist()
                     for n, rows in ev.items()}
            pred_f.write(json.dumps({"task": tname, "encoder": "none", "system": "S5c_char_neural", "resample": r,
                                     "C": 1.0, "preds": preds}) + "\n")
        for enc in args.encoder:
            st = Store(enc)
            for name, (view, pairs) in SYSTEMS[enc].items():
                xtr = st([v(e.text, e.form, view) for e in tr])
                xev = {n: st([v(e.text, e.form, view) for e in rows]) for n, rows in ev.items()}
                sx = st([v(e.text, e.form, view) for e in sel])
                tf = C.removal(st, list(ps[pairs].values()), C.K) if pairs else (lambda x: x)
                for r, sub in enumerate(subs):
                    c, preds = C.fit_eval(xtr[sub], ytr_full[sub], xev, None, sx, sel_y, tf)
                    pred_f.write(json.dumps({"task": tname, "encoder": enc, "system": name, "resample": r,
                                             "C": c, "preds": preds}) + "\n")
            pred_f.flush()
            print(tname, enc, "done", round(time.time() - t_start), "s", flush=True)
    pred_f.close()
    (out / "gold.json").write_text(json.dumps(gold))
    (out / f"timing.{args.part}.json").write_text(json.dumps({"seconds": round(time.time() - t_start, 1),
                                                             "encoders": args.encoder, "char": args.char}))


CONTRASTS = {"E1 S1T>S0": ("S1T_tarc_pairs", "S0_raw"), "E1 S1T vs S1": ("S1T_tarc_pairs", "S1_human"),
             "E2 S1 vs S3c": ("S1_human", "S3c_neural_pairs"), "E2 S3c vs S3": ("S3c_neural_pairs", "S3_dict_pairs"),
             "E3 S4c vs S4b": ("S4c_derom_neural", "S4b_derom_dict"), "E3 S1 vs S4c": ("S1_human", "S4c_derom_neural"),
             "E4 H1 S1>S0": ("S1_human", "S0_raw"), "E4 H2 S1>S2": ("S1_human", "S2_rule_pairs"),
             "E4 H3 S1>S3": ("S1_human", "S3_dict_pairs")}


def cmd_stats(args):
    from collections import defaultdict
    from .depstats import run as dep_run
    from .eval.metrics import macro_f1
    d = OUT / args.split
    gold = json.loads((d / "gold.json").read_text())
    merged = d / "predictions.jsonl"
    with open(merged, "w") as fo:
        for part in sorted(d.glob("predictions.*.jsonl")):
            fo.write(part.read_text())
    P = defaultdict(list)
    for l in open(merged):
        r = json.loads(l)
        P[(r["task"], r["encoder"], r["system"])].append(r["preds"])
    summ = {}
    for (t, e, s), pl in P.items():
        for n in gold[t]:
            summ.setdefault(f"{t}|{e}|{s}", {})[n] = 100 * float(np.mean([macro_f1(gold[t][n]["labels"], p[n]) for p in pl]))
    res = {}
    for enc_set, keys in (("three_encoders", ["e5-small", "labse", "xlmr-base"]), ("labse-st", ["labse-st"])):
        hy = {k: v for k, v in CONTRASTS.items() if (k.startswith("E4") == (enc_set == "labse-st"))}
        if enc_set == "labse-st":
            hy.update({k: v for k, v in CONTRASTS.items() if not k.startswith("E4")})
        res[enc_set] = dep_run(d, hy, args.n_boot, encoders=keys)
    (d / "stats.json").write_text(json.dumps({"summary": summ, "contrasts": res}, indent=1, default=float))
    for es, rr in res.items():
        for h, r in rr.items():
            c = r["cell_mean_joint"]
            tk = " ".join(f"{t.split('_')[0][:4]}{t[-4:]} {v['observed']:+.1f}[{v['lo']:+.1f},{v['hi']:+.1f}]"
                          for t, v in r["tasks"].items())
            print(f"{es:14s} {h:14s} {c['observed']:+.2f} [{c['lo']:+.2f},{c['hi']:+.2f}] | {tk}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--encoder", nargs="+", default=["e5-small", "labse", "xlmr-base"])
    e.add_argument("--threads", type=int, default=2)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--split", choices=["dev", "test"], required=True)
    ev.add_argument("--encoder", nargs="+", default=["e5-small", "labse", "xlmr-base", "labse-st"])
    ev.add_argument("--part", default="main", help="predictions.<part>.jsonl; stats merges all parts")
    ev.add_argument("--no-char", dest="char", action="store_false")
    s = sub.add_parser("stats")
    s.add_argument("--split", choices=["dev", "test"], required=True)
    s.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    {"encode": cmd_encode, "evaluate": cmd_evaluate, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    main()
