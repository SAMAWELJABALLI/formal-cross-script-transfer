"""Pilot experiment on cached embeddings: baselines vs FORMAL-SI, dev data only.

Usage: python -m formal_si.run --config configs/pilot.yaml

Saves runs/<name>/<encoder>/: config.yaml, metrics.jsonl (one row per method x setting x
resample), predictions.npz, and timing. Test rows are never read unless
``evaluate_split: test`` is set explicitly (confirmatory stage only).
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
from .data.schema import check_no_leakage
from .eval import metrics as M
from .methods.adapter import AdapterTrainer
from .views import apply_emoji, apply_view, form_of_text, view_text
from .methods.projections import (FormalSI, Identity, LEACE, LowRankMeanRemoval,
                                  MeanCentering, RandomRemoval, SplinceStyle)


class EmbeddingStore:
    def __init__(self, cache_dir: Path):
        self.emb = np.load(cache_dir / "emb.npy").astype(np.float64)
        texts = json.loads((cache_dir / "texts.json").read_text())
        self.index = {t: i for i, t in enumerate(texts)}
        self.meta = json.loads((cache_dir / "meta.json").read_text())

    def __call__(self, texts):
        missing = [t for t in texts if t not in self.index]
        if missing:
            raise KeyError(f"{len(missing)} texts not in cache, e.g. {missing[0][:60]!r}")
        return self.emb[[self.index[t] for t in texts]]


def build_pairs(examples, split, forms):
    """Meaning-preserving pairs: every pair of distinct forms inside one group."""
    by_group = defaultdict(dict)
    for e in examples:
        if e.split == split and e.form in forms:
            by_group[e.group].setdefault(e.form, e.text)
    a, b, fa, fb = [], [], [], []
    for g, fm in by_group.items():
        for f1, f2 in itertools.combinations(sorted(fm), 2):
            a.append(fm[f1])
            b.append(fm[f2])
            fa.append(f1)
            fb.append(f2)
    return a, b, fa, fb


def load_contrast(path, split, view="raw"):
    """jsonl rows: {text, paraphrase (another form of the same sentence, optional),
    negated, split}. Strings are mapped through the same input view as the examples."""
    if not path or not Path(path).exists():
        return [], [], []
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("split") == split and r.get("negated")]

    def v(t):
        return view_text(t, form_of_text(t), view)
    return ([v(r["text"]) for r in rows], [v(r.get("paraphrase") or r["text"]) for r in rows],
            [v(r["negated"]) for r in rows])


def fit_classifier(x, y, c):
    return LogisticRegression(C=c, max_iter=2000).fit(x, y)


def make_method(spec, seed):
    kind = spec["kind"]
    if kind == "raw":
        return Identity()
    if kind == "formal_si":
        return FormalSI(k=spec["k"], lam0=spec["lam0"], beta=spec["beta"])
    if kind == "leace":
        return LEACE()
    if kind == "splince":
        return SplinceStyle()
    if kind == "centering":
        return MeanCentering()
    if kind == "lsar":
        return LowRankMeanRemoval(r=spec.get("r", 1))
    if kind == "random":
        return RandomRemoval(k=spec["k"], seed=seed)
    if kind == "adapter":
        return AdapterTrainer(rank=spec.get("rank", 16), alpha=spec.get("alpha", 1.0),
                              gamma=spec.get("gamma", 1.0), seed=seed,
                              epochs=spec.get("epochs", 50))
    raise ValueError(kind)


def expand_grid(methods_cfg):
    out = []
    for m in methods_cfg:
        grid = {k: v for k, v in m.items() if isinstance(v, list)}
        fixed = {k: v for k, v in m.items() if not isinstance(v, list)}
        for combo in itertools.product(*grid.values()) if grid else [()]:
            out.append({**fixed, **dict(zip(grid.keys(), combo))})
    return out


def run(cfg: dict, examples, store: EmbeddingStore, out_dir: Path, view: str = "raw",
        method_label: str | None = None, append: bool = False, aux_examples=None):
    """``view`` maps the input strings (raw or deromanized); ``method_label`` overrides the
    method name in the log (e.g. derom_encode for raw-on-deromanized)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not append:
        (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    examples = apply_view(examples, view)
    check_no_leakage(examples)
    # Pair / concept data may come from other corpora than the task rows (pair_sources).
    aux = apply_view(aux_examples, view) if aux_examples is not None else examples
    ev = cfg.get("evaluate_split", "dev")
    task = cfg["task"]
    rows = [e for e in examples if e.task == task and e.label is not None]
    train = [e for e in rows if e.split == "train" and e.form in cfg["train_forms"]]
    evals = {name: [e for e in rows if e.split == ev and e.form in forms]
             for name, forms in cfg["eval_sets"].items()}
    pa, pb, _, _ = build_pairs(aux, "train", cfg["pair_forms"])
    pa_ev, pb_ev, fa_ev, fb_ev = build_pairs(aux, ev, cfg["pair_forms"])
    ca, cp, cn = load_contrast(cfg.get("contrast_file"), "train", view)
    ea, ep, en = load_contrast(cfg.get("contrast_file"), ev, view)
    concept_rows = [e for e in aux if e.split == "train"]  # all train rows, any task
    x_concept, z_concept = store([e.text for e in concept_rows]), [e.form for e in concept_rows]
    dialect_rows = [e for e in examples if e.split in ("train", ev)]
    log = open(out_dir / "metrics.jsonl", "a" if append else "w")
    pred_log = open(out_dir / "dev_predictions.jsonl", "a" if append else "w")
    if not append:  # dev gold labels and groups, once per eval set
        (out_dir / "dev_gold.json").write_text(json.dumps(
            {n: {"labels": [str(e.label) for e in r], "groups": [e.group for e in r],
                 "uids": [e.uid for e in r]} for n, r in evals.items()}))
    rng = np.random.default_rng(cfg.get("seed", 0))
    x_tr_full = store([e.text for e in train])
    y_tr_full = np.array([e.label for e in train])
    forms_tr_full = np.array([e.form for e in train])
    for spec in expand_grid(cfg["methods"]):
        for r in range(cfg.get("resamples", 5)):
            seed = int(rng.integers(1 << 31))
            sub = np.random.default_rng(seed).choice(len(train), int(0.9 * len(train)),
                                                     replace=False)
            x_tr, y_tr, f_tr = x_tr_full[sub], y_tr_full[sub], forms_tr_full[sub]
            t0 = time.time()
            m = make_method(spec, seed)
            kind = spec["kind"]
            if kind == "formal_si":
                m.fit(x_tr, y_tr, store(pa), store(pb),
                      store(ca) if ca else None, store(cn) if cn else None)
            elif kind in ("leace", "lsar", "centering"):
                m.fit(x_concept, z_concept)
            elif kind == "splince":
                m.fit(x_tr, f_tr, y_tr)
            elif kind == "adapter":
                m.fit(x_tr, y_tr, store(pa) if pa else None, store(pb) if pb else None,
                      store(ca) if ca else None, store(cn) if cn else None)
            else:
                m.fit(x_tr)

            def tf(x, forms=None):
                return m.transform(x, forms) if kind == "centering" else m.transform(x)

            h_tr = tf(x_tr, f_tr)
            label = (method_label if (method_label and kind == "raw")
                     else (f"derom+{kind}" if view == "deromanized" else kind))
            res = {"method": label, "view": view, **{k: v for k, v in spec.items() if k != "kind"},
                   "resample": r, "seed": seed, "encoder": store.meta["model_id"]}
            best = None
            for c in cfg.get("C", [0.1, 1.0, 10.0]):
                clf = fit_classifier(h_tr, y_tr, c)
                scores, preds = {}, {}
                for name, ev_rows in evals.items():
                    if not ev_rows:
                        continue
                    hx = tf(store([e.text for e in ev_rows]), [e.form for e in ev_rows])
                    pred = clf.predict(hx)
                    preds[name] = [str(p) for p in pred]
                    scores[f"f1_{name}"] = M.macro_f1([e.label for e in ev_rows], pred)
                    scores[f"consistency_{name}"] = M.cross_form_consistency(
                        pred, [e.group for e in ev_rows])
                key = scores.get(f"f1_{cfg['select_on']}", -1)
                if best is None or key > best[0]:
                    best = (key, c, scores, clf, preds)
            _, c, scores, clf, preds = best
            res.update(scores, C=c)
            # Dev predictions of the selected C, for the grouped paired bootstrap.
            pred_log.write(json.dumps({**{k: v for k, v in res.items()
                                          if k not in ("encoder",)}, "preds": preds}) + "\n")
            if pa_ev:
                res["invariance"] = M.invariance_score(tf(store(pa_ev), fa_ev),
                                                       tf(store(pb_ev), fb_ev))
            if ea:
                def forms_of(ts):
                    return [form_of_text(t) for t in ts]
                anc, par, neg = (tf(store(t), forms_of(t)) for t in (ea, ep, en))
                res["negation_auc"] = M.sensitivity_auc(anc, par, neg)
                res["negation_flip_rate"] = M.flip_rate(clf.predict(anc), clf.predict(neg))
            if cfg.get("dialect_probe") and len({e.dialect for e in dialect_rows}) > 1:
                dtr = [e for e in dialect_rows if e.split == "train"]
                if len(dtr) > 6000:  # fixed subsample keeps the probe cheap (same rows every fit)
                    keep = np.random.default_rng(0).choice(len(dtr), 6000, replace=False)
                    dtr = [dtr[i] for i in sorted(keep)]
                dev = [e for e in dialect_rows if e.split == ev]
                probe = fit_classifier(tf(store([e.text for e in dtr]), [e.form for e in dtr]),
                                       [e.dialect for e in dtr], 1.0)
                res["dialect_f1"] = M.macro_f1([e.dialect for e in dev], probe.predict(
                    tf(store([e.text for e in dev]), [e.form for e in dev])))
            res["fit_seconds"] = round(time.time() - t0, 3)
            log.write(json.dumps(res) + "\n")
            log.flush()
    log.close()
    pred_log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    examples = apply_emoji(load_jsonl(Path(cfg["examples"])), cfg.get("strip_emoji_sources"))
    all_ex = examples
    aux = None
    if cfg.get("sources"):
        examples = [e for e in examples if e.source in set(cfg["sources"])]
        if cfg.get("pair_sources"):
            aux = [e for e in all_ex
                   if e.source in set(cfg["pair_sources"]) | set(cfg["sources"])]
    for enc in cfg["encoders"]:
        store = EmbeddingStore(Path(cfg["cache"]) / enc)
        out = Path(cfg["out"]) / enc
        run(cfg, examples, store, out, aux_examples=aux)
        if "deromanized" in cfg.get("input_views", []):
            # Deromanize-then-encode baseline: raw frozen embedding of canonicalised input.
            # derom_methods (optional): methods also run on the deromanized view, logged as
            # "derom+<kind>", outside the pre-registered gates (analysis of point 5 only).
            run({**cfg, "methods": [{"kind": "raw"}] + cfg.get("derom_methods", [])},
                examples, store, out,
                view="deromanized", method_label="derom_encode", append=True, aux_examples=aux)


if __name__ == "__main__":
    main()
