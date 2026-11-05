"""P2 diagnostics and the explanation analysis for the pilot (dev split only).

For each cached encoder:
  * script probe: logistic regression Arabic-script vs Arabizi (train rows -> dev rows);
  * invariance of raw embeddings on human dev pairs (NArabizi + TArC);
  * subspace overlap: share of the Arabic-script sentiment direction (class-mean difference,
    and the LR weight vector) that lies in the top-k invariance subspace (top eigenvectors
    of S_inv), against the k/d share expected for a random direction;
  * transfer of task directions: cosine between the LR weight trained on Arabic-script rows
    and the one trained on Arabizi rows of the same corpus.
Usage: python -m formal_si.diagnostics --config configs/pilot.yaml --out reports/pilot/diagnostics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

from .data.audit import load_jsonl
from .eval import metrics as M
from .methods.projections import difference_scatter
from .run import EmbeddingStore, build_pairs
from .views import apply_emoji


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--out", default="reports/pilot/diagnostics.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ex = [e for e in apply_emoji(load_jsonl(Path(cfg["examples"])), cfg.get("strip_emoji_sources"))
          if e.split in ("train", "dev")]
    pair_ex = [e for e in ex if e.source in ("narabizi", "tarc")]
    out = {}
    for enc in cfg["encoders"]:
        st = EmbeddingStore(Path(cfg["cache"]) / enc)
        d = st.emb.shape[1]
        res = {}
        # Script probe
        sc = [e for e in ex if e.form in ("arabic", "arabizi") and e.source in ("narabizi", "tsac", "myc")]
        tr = [e for e in sc if e.split == "train"]
        dv = [e for e in sc if e.split == "dev"]
        rng = np.random.default_rng(0)
        tr = [tr[i] for i in rng.choice(len(tr), min(8000, len(tr)), replace=False)]
        p = LogisticRegression(C=1.0, max_iter=2000).fit(st([e.text for e in tr]), [e.form for e in tr])
        res["script_probe_acc_dev"] = float(np.mean(p.predict(st([e.text for e in dv])) == np.array([e.form for e in dv])))
        # Invariance subspace from train human pairs
        pa, pb, _, _ = build_pairs(pair_ex, "train", ["arabic", "arabizi", "codeswitch"])
        s_inv = difference_scatter(st(pa), st(pb))
        vals, vecs = np.linalg.eigh(s_inv)
        vecs = vecs[:, np.argsort(vals)[::-1]]
        pa_d, pb_d, _, _ = build_pairs(pair_ex, "dev", ["arabic", "arabizi", "codeswitch"])
        res["invariance_raw_dev_pairs"] = float(M.invariance_score(st(pa_d), st(pb_d)))
        res["s_inv_top16_variance_share"] = float(np.sort(vals)[::-1][:16].sum() / vals.sum())
        per = {}
        for src in ("narabizi", "tsac", "myc"):
            rows = [e for e in ex if e.source == src and e.task == "sentiment" and e.split == "train"]
            r = {}
            w = {}
            for form in ("arabic", "arabizi"):
                rf = [e for e in rows if e.form == form]
                x, y = st([e.text for e in rf]), np.array([e.label for e in rf])
                clf = LogisticRegression(C=1.0, max_iter=3000).fit(x, y)
                w[form] = unit(clf.coef_[0] if clf.classes_[1] == "pos" else -clf.coef_[0])
                if form == "arabic":
                    mdiff = unit(x[y == "pos"].mean(0) - x[y == "neg"].mean(0))
            r["cos_w_arabic_vs_w_arabizi"] = float(w["arabic"] @ w["arabizi"])
            for k in (4, 16, 64):
                U = vecs[:, :k]
                r[f"share_meandiff_in_inv_top{k}"] = float(np.sum((U.T @ mdiff) ** 2))
                r[f"share_w_arabic_in_inv_top{k}"] = float(np.sum((U.T @ w["arabic"]) ** 2))
                r[f"share_w_arabizi_in_inv_top{k}"] = float(np.sum((U.T @ w["arabizi"]) ** 2))
                r[f"random_expected_top{k}"] = k / d
            per[src] = r
        res["per_corpus"] = per
        out[enc] = res
        print(enc, json.dumps(res, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
