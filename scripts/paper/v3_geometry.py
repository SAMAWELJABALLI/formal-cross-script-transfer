"""V3 geometry and visualisation data (TRAIN pairs + DEV rows only; the test split is not read).

For every encoder and pair source (none, human, rule, dict, neural, TArC) the top-16 uncentered pair-difference
subspace is estimated from the same 998 training pairs used in the experiments (confirm / confirm2) and applied to
held-out dev rows. Outputs reports/v3/geometry_v3.json and reports/v3/viz_coords.json:
  - identity check ||P x||^2 = ||x||^2 - ||U^T x||^2 (max abs deviation)
  - script probe accuracy (GroupKFold logistic regression) on NArabizi dev and on TSAC dev
  - paired cosine distance and pair retrieval@1 on NArabizi dev (Arabizi -> its own Arabic-script row)
  - held-out residual pair energy  sum||P D||^2 / sum||D||^2
  - sentiment-direction energy in the removed subspace (Arabic-script train class-mean difference)
  - prediction-agreement certificate on NArabizi dev sentiment pairs (Proposition 4 of the paper)
  - fixed-basis PCA coordinates (basis fitted on raw dev embeddings only), t-SNE and UMAP coordinates
Level B only (needs the rebuilt corpus and the embedding caches):  python scripts/paper/v3_geometry.py
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupKFold

from formal_si import confirm as C
from formal_si import confirm2 as C2
from formal_si.eval.metrics import macro_f1

from _paths import REPORTS_V3

OUT = REPORTS_V3
ENCS = ["e5-small", "labse", "xlmr-base"]
SOURCES = ["none", "human", "rule", "dict", "neural", "tarc"]
K = 16
SEED = 0


def basis(st, pairs, k=K):
    a = st([p[0] for p in pairs]); b = st([p[1] for p in pairs])
    D = a - b
    S = D.T @ D / len(D)
    vals, vecs = np.linalg.eigh(S)
    o = np.argsort(vals)[::-1]
    return vals[o], vecs[:, o[:k]]


def proj(X, U):
    return X if U is None else X - (X @ U) @ U.T


def probe(X, y, groups):
    acc = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(X[tr], y[tr])
        acc.append((clf.predict(X[te]) == y[te]).mean())
    return 100 * float(np.mean(acc))


def cos_rows(A, B):
    return np.sum(A * B, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))


def retrieval(Za, Zb):
    na = Za / np.linalg.norm(Za, axis=1, keepdims=True)
    nb = Zb / np.linalg.norm(Zb, axis=1, keepdims=True)
    sim = nb @ na.T
    return 100 * float(np.mean(np.argmax(sim, 1) == np.arange(len(Zb))))


def main():
    t0 = time.time()
    ex = C.load(include_test=False)
    trdev = [e for e in ex if e.split != "test"]
    v = C2.Views(trdev)
    _, ps = C2.pair_sets2(trdev)
    pairsets = {s: list(ps[s].values()) for s in SOURCES if s != "none"}
    # NArabizi dev pairs (all tasks) for pair metrics; sentiment dev pairs for classifier analysis
    g = defaultdict(dict)
    for e in ex:
        if e.source == "narabizi" and e.split == "dev":
            g[e.group].setdefault(e.form, e.text)
            if e.task == "sentiment":
                g[e.group]["label"] = e.label
    pg = sorted(k for k, fm in g.items() if "arabic" in fm and "arabizi" in fm)
    sg = [k for k in pg if "label" in g[k]]
    rows = [e for e in ex if e.source == "narabizi" and e.task == "sentiment" and e.label]
    tr = [e for e in rows if e.split == "train" and e.form == "arabic"]
    sel = [e for e in rows if e.split == "dev" and e.form == "arabic"]
    ytr = np.array([e.label for e in tr]); ysel = [e.label for e in sel]
    # TSAC dev sample (not parallel) for out-of-corpus script probe
    rng = np.random.default_rng(SEED)
    ts = [e for e in ex if e.source == "tsac" and e.task == "sentiment" and e.split == "dev" and e.form in ("arabic", "arabizi")]
    ts_ar = [e for e in ts if e.form == "arabic"]; ts_az = [e for e in ts if e.form == "arabizi"]
    ts_ar = [ts_ar[i] for i in sorted(rng.choice(len(ts_ar), 300, replace=False))]
    ts_az = [ts_az[i] for i in sorted(rng.choice(len(ts_az), 300, replace=False))]
    res = {"meta": {"k": K, "n_pairs_train": {s: len(p) for s, p in pairsets.items()}, "n_dev_pairs": len(pg),
                    "n_dev_sent_pairs": len(sg), "tsac_sample": [len(ts_ar), len(ts_az)], "seed": SEED,
                    "sample_ids": {"narabizi_dev_sentiment_groups": sg, "narabizi_dev_pair_groups": pg,
                                   "tsac_dev_uids": [e.uid for e in ts_ar + ts_az]}},
           "enc": {}}
    viz = {"groups": sg, "labels": [g[k]["label"] for k in sg], "enc": {}}
    for enc in ENCS:
        st = C2.Store(enc)
        A = st([g[k]["arabic"] for k in pg]); B = st([g[k]["arabizi"] for k in pg])
        As = st([g[k]["arabic"] for k in sg]); Bs = st([g[k]["arabizi"] for k in sg])
        TA = st([e.text for e in ts_ar]); TB = st([e.text for e in ts_az])
        Xtr = st([e.text for e in tr]); Xsel = st([e.text for e in sel])
        mu = Xtr[ytr == "pos"].mean(0) - Xtr[ytr == "neg"].mean(0)
        Us, spectra = {}, {}
        for s in SOURCES:
            if s == "none":
                Us[s] = None
            else:
                vals, U = basis(st, pairsets[s])
                Us[s] = U
                spectra[s] = (vals[:64] / vals.sum()).tolist()
        out = {"spectra_frac": spectra}
        # fixed PCA basis from RAW dev embeddings (sentiment pairs, both scripts)
        R = np.vstack([As, Bs]); mean = R.mean(0)
        _, sv, Vt = np.linalg.svd(R - mean, full_matrices=False)
        V3 = Vt[:3].T
        # interpretable fixed axes: leading human pair-difference direction and the Arabic-train sentiment direction
        u1 = Us["human"][:, 0]
        if np.mean(As @ u1) < np.mean(Bs @ u1):
            u1 = -u1
        sp = mu - (mu @ u1) * u1; sp /= np.linalg.norm(sp)
        Zh = proj(np.vstack([As, Bs]), Us["human"]); Zh = Zh - Zh.mean(0)
        r3 = np.linalg.svd(Zh, full_matrices=False)[2][0]
        r3 = r3 - (r3 @ u1) * u1 - (r3 @ sp) * sp; r3 /= np.linalg.norm(r3)
        evr = (sv ** 2 / np.sum(sv ** 2))[:3].tolist()
        venc = {"pca_raw_evr": evr, "sources": {}}
        for s in SOURCES:
            U = Us[s]
            ZA, ZB = proj(A, U), proj(B, U)
            m = {}
            # identity check
            if U is not None:
                lhs = np.sum(ZA ** 2, 1); rhs = np.sum(A ** 2, 1) - np.sum((A @ U) ** 2, 1)
                m["identity_max_abs_dev"] = float(np.max(np.abs(lhs - rhs)))
                m["mean_removed_energy"] = float(np.mean(np.sum((A @ U) ** 2, 1)))
                m["mean_removed_energy_arabizi"] = float(np.mean(np.sum((B @ U) ** 2, 1)))
                m["sent_dir_energy_removed"] = float(np.sum((U.T @ mu) ** 2) / np.sum(mu ** 2))
            m["mean_sq_norm_raw"] = float(np.mean(np.sum(A ** 2, 1)))
            y = np.array([0] * len(ZA) + [1] * len(ZB))
            grp = np.array(list(range(len(ZA))) * 2)
            m["script_probe_narabizi"] = probe(np.vstack([ZA, ZB]), y, grp)
            TZA, TZB = proj(TA, U), proj(TB, U)
            yt = np.array([0] * len(TZA) + [1] * len(TZB))
            m["script_probe_tsac"] = probe(np.vstack([TZA, TZB]), yt, np.arange(len(yt)))
            m["paired_cos_dist"] = float(np.mean(1 - cos_rows(ZA, ZB)))
            idx = np.random.default_rng(SEED).permutation(len(ZB))
            m["unpaired_cos_dist"] = float(np.mean(1 - cos_rows(ZA, ZB[idx])))
            m["retrieval_at1"] = retrieval(ZA, ZB)
            D = A - B
            m["heldout_residual_energy"] = float(np.sum(proj(D, U) ** 2) / np.sum(D ** 2))
            # classifier (protocol: C chosen on in-script dev; full train, no resampling) + certificate
            best = None
            for c in C.CS:
                clf = LogisticRegression(C=c, max_iter=3000).fit(proj(Xtr, U), ytr)
                f = macro_f1(ysel, clf.predict(proj(Xsel, U)))
                if best is None or f > best[0]:
                    best = (f, c, clf)
            _, c, clf = best
            w = clf.coef_[0]; b0 = clf.intercept_[0]
            w_p = proj(w[None], U)[0]
            ZAs, ZBs = proj(As, U), proj(Bs, U)
            sa = ZAs @ w + b0; sb = ZBs @ w + b0
            bound = np.linalg.norm(w_p) * np.linalg.norm(ZAs - ZBs, axis=1)
            ys = np.array(viz["labels"])
            pa = np.where(sa > 0, clf.classes_[1], clf.classes_[0]); pb = np.where(sb > 0, clf.classes_[1], clf.classes_[0])
            m["clf"] = {"C": c, "w_norm": float(np.linalg.norm(w)), "w_in_complement": float(np.linalg.norm(w_p) / np.linalg.norm(w)),
                        "agreement": 100 * float(np.mean(np.sign(sa) == np.sign(sb))),
                        "certified": 100 * float(np.mean(np.abs(sa) > bound)),
                        "mean_abs_score_gap": float(np.mean(np.abs(sa - sb))), "mean_bound": float(np.mean(bound)),
                        "tightness": float(np.mean(np.abs(sa - sb) / bound)),
                        "f1_arabic_dev": 100 * macro_f1(list(ys), list(pa)), "f1_arabizi_dev": 100 * macro_f1(list(ys), list(pb)),
                        "mean_margin_arabic": float(np.mean(np.abs(sa)))}
            out[s] = m
            # visualisation coordinates
            ZsA, ZsB = proj(As, U), proj(Bs, U)
            Z = np.vstack([ZsA, ZsB])
            ax2 = np.stack([Z @ u1, Z @ sp], 1)
            ax3 = np.stack([Z @ u1, Z @ sp, Z @ r3], 1)
            yv = np.concatenate([np.array(viz["labels"])] * 2)
            def dprime(v_, m_):
                a_, b_ = v_[m_], v_[~m_]
                return float((a_.mean() - b_.mean()) / np.sqrt((a_.var() + b_.var()) / 2))
            scr = np.array([True] * len(ZsA) + [False] * len(ZsB))
            m.setdefault("axes", {})
            m["axes"] = {"script_d_on_u1": dprime(ax2[:, 0], scr), "sent_d_on_sdir": dprime(ax2[:, 1], yv == "pos"),
                         "sent_d_on_sdir_arabizi": dprime(ax2[len(ZsA):, 1], yv[len(ZsA):] == "pos"),
                         "sent_d_on_sdir_arabic": dprime(ax2[:len(ZsA), 1], yv[:len(ZsA)] == "pos")}
            vs = {"pca3": ((Z - mean) @ V3).tolist(), "axes2": ax2.tolist(), "axes3": ax3.tolist(),
                  "pca_fixed_var_captured": float(np.sum(((Z - Z.mean(0)) @ V3) ** 2) / np.sum((Z - Z.mean(0)) ** 2))}
            if enc == "e5-small" or s in ("none", "human", "rule", "dict"):
                vs["tsne"] = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED, max_iter=1000,
                                  metric="cosine").fit_transform(Z).tolist()
                import umap
                vs["umap"] = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(Z).tolist()
            if enc == "e5-small":
                TZ = np.vstack([proj(TA[:150], U), proj(TB[:150], U)])
                if s == "none":
                    tmean = TZ.mean(0); _, _, tVt = np.linalg.svd(TZ - tmean, full_matrices=False); venc["tsac_V"] = tVt[:3].T
                vs["tsac_pca3"] = ((TZ - tmean) @ venc["tsac_V"]).tolist()
            venc["sources"][s] = vs
            print(enc, s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k != "clf"},
                  {k: round(v, 3) for k, v in m["clf"].items()}, round(time.time() - t0), "s", flush=True)
        venc.pop("tsac_V", None)
        # principal angles human vs others
        out["overlap_with_human"] = {}
        for s in SOURCES[2:]:
            sv_ = np.clip(np.linalg.svd(Us["human"].T @ Us[s], compute_uv=False), 0, 1)
            out["overlap_with_human"][s] = {"mean_cos2": float(np.mean(sv_ ** 2)), "angles_deg": np.degrees(np.arccos(sv_)).tolist()}
        res["enc"][enc] = out
        viz["enc"][enc] = venc
    res["meta"]["seconds"] = round(time.time() - t0, 1)
    (OUT / "geometry_v3.json").write_text(json.dumps(res))
    (OUT / "viz_coords.json").write_text(json.dumps(viz))
    print("done", res["meta"]["seconds"])


if __name__ == "__main__":
    main()
