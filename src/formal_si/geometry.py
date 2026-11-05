"""Representation-geometry analyses for the revised manuscript (train/dev data only; no test rows).

Outputs reports/geometry/geometry.json and reports/geometry/pca_*.json with:
  * spectra of the uncentered pair-difference scatter for human / rule / dictionary pairs (and TArC human pairs);
  * effective rank, participation ratio, top-k energy share;
  * subspace similarity (principal angles, projection overlap ||U1^T U2||_F^2 / k) between pair sources and
    across pair subsamples (stability);
  * held-out pair-difference energy removed on NArabizi dev and TArC dev pairs;
  * linear script decodability before/after projection;
  * task-script overlap E_k with the human subspace and principal angles to the task direction;
  * timing of the eigendecomposition and of the projection.
Usage: python -m formal_si.geometry
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from .confirm import Store, load, pair_sets
from .methods.projections import between_class_scatter
from .text.normalize import arabic_script_ratio

ENCS = ["e5-small", "labse", "xlmr-base"]
K = 16
OUT = Path("reports/geometry")


def scatter(st, pairs):
    D = st([a for a, _ in pairs]) - st([b for _, b in pairs])
    return D.T @ D / len(D), D


def top(S, k):
    vals, vecs = np.linalg.eigh(S)
    o = np.argsort(vals)[::-1]
    return vals[o], vecs[:, o[:k]]


def overlap(U1, U2):
    s = np.linalg.svd(U1.T @ U2, compute_uv=False)
    s = np.clip(s, 0, 1)
    return float(np.sum(s ** 2) / U1.shape[1]), np.degrees(np.arccos(s)).tolist()


def eff_rank(vals):
    p = np.clip(vals, 0, None)
    p = p / p.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum())), float(1.0 / (p ** 2).sum())


def tarc_pairs(ex, split, n=None, seed=0):
    g = defaultdict(dict)
    for e in ex:
        if e.source == "tarc" and e.split == split:
            g[e.group].setdefault(e.form, e.text)
    ps = sorted((fm["codeswitch"], fm["arabizi"]) for fm in g.values()
                if "codeswitch" in fm and "arabizi" in fm and arabic_script_ratio(fm["codeswitch"]) >= 1.0)
    if n:
        idx = np.random.default_rng(seed).choice(len(ps), n, replace=False)
        ps = [ps[i] for i in sorted(idx)]
    return ps


def nar_dev_pairs(ex):
    g = defaultdict(dict)
    for e in ex:
        if e.source == "narabizi" and e.split == "dev":
            g[e.group].setdefault(e.form, e.text)
    return [(fm["arabic"], fm["arabizi"]) for fm in g.values() if "arabic" in fm and "arabizi" in fm]


def main():
    ex = load(include_test=False)
    groups, ps = pair_sets(ex)
    sources = {s: [ps[s][g] for g in groups] for s in ("human", "rule", "dict")}
    sources["tarc_human"] = tarc_pairs(ex, "train", n=len(groups))
    held = {"narabizi_dev": nar_dev_pairs(ex), "tarc_dev": tarc_pairs(ex, "dev")}
    res = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for enc in ENCS:
        st = Store(enc)
        r = {"spectra": {}, "subspaces": {}, "heldout_energy_removed": {}, "stability": {}}
        U = {}
        for s, pairs in sources.items():
            t0 = time.time()
            S, D = scatter(st, pairs)
            vals, Uk = top(S, K)
            r.setdefault("timing_eigh_s", {})[s] = round(time.time() - t0, 4)
            U[s] = Uk
            er, pr = eff_rank(vals)
            tr = vals.sum()
            r["spectra"][s] = {"eig_normalised": (vals[:100] / tr).tolist(), "trace": float(tr),
                               "top4_share": float(vals[:4].sum() / tr), "top16_share": float(vals[:16].sum() / tr),
                               "top64_share": float(vals[:64].sum() / tr), "effective_rank": er, "participation_ratio": pr,
                               "mean_pair_sq_distance": float((D ** 2).sum(1).mean())}
        for a in sources:
            for b in sources:
                if a < b:
                    ov, ang = overlap(U[a], U[b])
                    r["subspaces"][f"{a}|{b}"] = {"overlap": ov, "angles_deg": ang}
        d = st.parts[0][0].shape[1]
        r["random_overlap_expected"] = K / d
        # held-out energy removed and pair cosine
        for hname, hp in held.items():
            Da = st([a for a, _ in hp])
            Db = st([b for _, b in hp])
            D = Da - Db
            r["heldout_energy_removed"][hname] = {}
            for s, Uk in U.items():
                PD = D - (D @ Uk) @ Uk.T
                pa, pb = Da - (Da @ Uk) @ Uk.T, Db - (Db @ Uk) @ Uk.T
                cos_after = (pa * pb).sum(1) / (np.linalg.norm(pa, axis=1) * np.linalg.norm(pb, axis=1))
                r["heldout_energy_removed"][hname][s] = {
                    "energy_removed": float(1 - (PD ** 2).sum() / (D ** 2).sum()),
                    "pair_cos_before": float((Da * Db).sum(1).mean()), "pair_cos_after": float(cos_after.mean()),
                    "n_pairs": len(hp)}
        # stability across subsamples of human pairs
        hp = sources["human"]
        rng = np.random.default_rng(0)
        Sfull, _ = scatter(st, hp)
        _, Ufull = top(Sfull, K)
        for n in (50, 100, 250, 500):
            ovs = []
            for _ in range(10):
                idx = rng.choice(len(hp), n, replace=False)
                _, Us = top(scatter(st, [hp[i] for i in idx])[0], K)
                ovs.append(overlap(Us, Ufull)[0])
            r["stability"][str(n)] = {"mean": float(np.mean(ovs)), "sd": float(np.std(ovs))}
        perm = rng.permutation(len(hp))
        _, U1 = top(scatter(st, [hp[i] for i in perm[:len(hp) // 2]])[0], K)
        _, U2 = top(scatter(st, [hp[i] for i in perm[len(hp) // 2:]])[0], K)
        r["stability"]["split_half"] = overlap(U1, U2)[0]
        # script decodability before / after (train -> dev), NArabizi+TSAC+MYC Arabic vs Arabizi
        sc = [e for e in ex if e.form in ("arabic", "arabizi") and e.source in ("narabizi", "tsac", "myc")]
        tr = [e for e in sc if e.split == "train"]
        tr = [tr[i] for i in np.random.default_rng(0).choice(len(tr), 6000, replace=False)]
        dv = [e for e in sc if e.split == "dev"]
        xtr, xdv = st([e.text for e in tr]), st([e.text for e in dv])
        ytr, ydv = [e.form for e in tr], np.array([e.form for e in dv])
        r["script_probe"] = {}
        for s in ("none", "human", "rule", "dict"):
            P = (lambda x: x) if s == "none" else (lambda x, Uk=U[s]: x - (x @ Uk) @ Uk.T)
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(P(xtr), ytr)
            r["script_probe"][s] = float((clf.predict(P(xdv)) == ydv).mean())
        # task-script overlap
        r["task_overlap"] = {}
        for tname, task, src in [("narabizi_sent", "sentiment", "narabizi"), ("tsac_sent", "sentiment", "tsac"),
                                 ("myc_sent", "sentiment", "myc"), ("narabizi_topic", "topic", "narabizi")]:
            rows = [e for e in ex if e.source == src and e.task == task and e.label and e.split == "train"]
            out = {}
            for form in ("arabic", "arabizi"):
                rf = [e for e in rows if e.form == form]
                x, y = st([e.text for e in rf]), np.array([e.label for e in rf])
                St = between_class_scatter(x, y)
                vals_t, Ut = top(St, max(1, len(set(y)) - 1))
                out[form] = {f"E_{K}_human": float(np.trace(U["human"].T @ St @ U["human"]) / np.trace(St)),
                             f"E_{K}_rule": float(np.trace(U["rule"].T @ St @ U["rule"]) / np.trace(St)),
                             "angles_task_to_human_deg": overlap(Ut, U["human"][:, :K])[1]}
            out["random_expected"] = K / d
            r["task_overlap"][tname] = out
        # projection cost
        X = np.random.default_rng(0).standard_normal((10000, d))
        t0 = time.time()
        _ = X - (X @ U["human"]) @ U["human"].T
        r["timing_project_10k_s"] = round(time.time() - t0, 4)
        res[enc] = r
        # PCA data for the small-multiples figure (NArabizi dev pairs)
        hp_dev = held["narabizi_dev"]
        Da, Db = st([a for a, _ in hp_dev]), st([b for _, b in hp_dev])
        Uh = U["human"]
        pcs = {}
        for name, (A, B) in {"before": (Da, Db), "after_human": (Da - (Da @ Uh) @ Uh.T, Db - (Db @ Uh) @ Uh.T),
                             "after_rule": (Da - (Da @ U["rule"]) @ U["rule"].T, Db - (Db @ U["rule"]) @ U["rule"].T)}.items():
            p = PCA(n_components=3, random_state=0).fit(np.vstack([A, B]))
            pcs[name] = {"ar": p.transform(A).tolist(), "az": p.transform(B).tolist(),
                         "explained": p.explained_variance_ratio_.tolist()}
        (OUT / f"pca_{enc}.json").write_text(json.dumps(pcs))
        print(enc, "done", flush=True)
    (OUT / "geometry.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
