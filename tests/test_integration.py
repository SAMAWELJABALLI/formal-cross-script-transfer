"""End-to-end smoke test of run.py + decide.py on a SYNTHETIC embedding cache.

It checks that the pipeline wires together and writes its artefacts. The numbers it
produces mean nothing about real data and must never be reported as results."""

import json

import numpy as np
import yaml

from formal_si.data.schema import Example
from formal_si.decide import decide
from formal_si.run import EmbeddingStore, run


def test_pipeline_smoke(tmp_path):
    rng = np.random.default_rng(0)
    d, n_groups = 16, 120
    examples, texts, embs = [], [], []
    for g in range(n_groups):
        label = "pos" if g % 2 else "neg"
        dialect = ["TN", "DZ", "MA"][g % 3]
        split = "train" if g < 80 else "dev"
        base = rng.normal(0, 0.3, d)
        base[0] += 1.5 if label == "pos" else -1.5
        base[3] += {"TN": 1, "DZ": -1, "MA": 0}[dialect]
        for form, shift in (("arabic", 2.0), ("arabizi", -2.0)):
            t = f"{form}-{g}"
            v = base.copy()
            v[1] += shift
            examples.append(Example(uid=t, text=t, group=f"g{g}", form=form, dialect=dialect,
                                    source="synthetic", task="sentiment", label=label,
                                    split=split))
            texts.append(t)
            embs.append(v)
    contrast = tmp_path / "contrast.jsonl"
    with open(contrast, "w") as f:
        for g in range(n_groups):
            neg = f"neg-arabizi-{g}"
            v = embs[2 * g + 1].copy()
            v[0] *= -1
            texts.append(neg)
            embs.append(v)
            f.write(json.dumps({"text": f"arabizi-{g}", "paraphrase": f"arabic-{g}",
                                "negated": neg, "split": "train" if g < 80 else "dev"}) + "\n")
    cache = tmp_path / "cache" / "toy"
    cache.mkdir(parents=True)
    np.save(cache / "emb.npy", np.array(embs, dtype=np.float16))
    (cache / "texts.json").write_text(json.dumps(texts))
    (cache / "meta.json").write_text(json.dumps({"model_id": "synthetic"}))
    cfg = yaml.safe_load(open("configs/pilot.yaml"))
    cfg.update(task="sentiment", resamples=2, contrast_file=str(contrast),
               methods=[{"kind": "raw"}, {"kind": "leace"}, {"kind": "centering"},
                        {"kind": "splince"}, {"kind": "lsar", "r": 1},
                        {"kind": "formal_si", "k": [1, 2], "lam0": [0.5], "beta": [1.0]}])
    out = tmp_path / "runs" / "toy"
    run(cfg, examples, EmbeddingStore(cache), out)
    rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
    assert len(rows) == 2 * 7  # 5 single settings + 2 FORMAL-SI settings
    assert {"f1_cross_script", "negation_auc", "invariance", "dialect_f1"} <= set(rows[-1])
    text, facts = decide(tmp_path / "runs")
    assert "Verdict on dev data" in text and "toy" in facts
