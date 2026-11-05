"""Encode every unique string once with a frozen encoder and cache the embeddings.

Usage: python -m formal_si.encode --examples data/examples.jsonl --encoder e5-small labse
Writes cache/<encoder>/emb.npy (float16, L2-normalised), texts.json and meta.json
(model id, revision, pooling, runtime, sentences/s, peak RSS).
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

ENCODERS = {
    # short name: (HF id, text prefix, pooling)
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", "mean"),
    "labse": ("sentence-transformers/LaBSE", "", "cls"),
    "xlmr-base": ("FacebookAI/xlm-roberta-base", "", "mean"),
}

# Hugging Face revisions of the reported runs. The original runs passed no revision and therefore loaded the
# Hub's `main`, which resolved to these commits (recorded in the confirmatory protocol). They are pinned here
# for the public release so that a re-run cannot silently pick up a newer model version.
REVISIONS = {
    "intfloat/multilingual-e5-small": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    "sentence-transformers/LaBSE": "836121a0533e5664b21c7aacc5d22951f2b8b25b",
    "FacebookAI/xlm-roberta-base": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
}


def pool(hidden: torch.Tensor, mask: torch.Tensor, how: str) -> torch.Tensor:
    if how == "cls":
        out = hidden[:, 0]
    elif how == "mean":
        m = mask.unsqueeze(-1).to(hidden.dtype)
        out = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
    else:
        raise ValueError(how)
    return torch.nn.functional.normalize(out, dim=-1)


@torch.inference_mode()
def encode_texts(texts: list[str], model_id: str, prefix: str, pooling: str,
                 batch_size: int = 64, max_length: int = 128, device: str = "cpu",
                 revision: str | None = None) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    revision = revision or REVISIONS.get(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision).to(device).eval()
    order = np.argsort([len(t) for t in texts])  # length bucketing for speed
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for s in range(0, len(texts), batch_size):
        idx = order[s:s + batch_size]
        batch = tok([prefix + texts[i] for i in idx], padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt").to(device)
        hidden = model(**batch).last_hidden_state
        out[idx] = pool(hidden, batch["attention_mask"], pooling).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="data/examples.jsonl")
    ap.add_argument("--extra-texts", nargs="*", default=[],
                    help="jsonl files with {'text': ...} rows, e.g. the negation contrast set")
    ap.add_argument("--config", default=None,
                    help="pilot yaml: encode every string its input views and contrast set "
                         "need (emoji-stripped, deromanized), train + evaluate_split only")
    ap.add_argument("--encoder", nargs="+", default=["e5-small"])
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    texts = []
    if args.config:
        import yaml
        from .data.audit import load_jsonl
        from .views import all_texts
        cfg = yaml.safe_load(open(args.config))
        texts = all_texts(cfg, load_jsonl(Path(cfg["examples"])))
    else:
        for path in [args.examples, *args.extra_texts]:
            with open(path, encoding="utf-8") as f:
                texts += [json.loads(l)["text"] for l in f if l.strip()]
    texts = sorted(set(texts))
    for name in args.encoder:
        model_id, prefix, pooling = ENCODERS[name]
        t0 = time.time()
        emb = encode_texts(texts, model_id, prefix, pooling)
        dt = time.time() - t0
        d = Path(args.cache) / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "emb.npy", emb.astype(np.float16))
        (d / "texts.json").write_text(json.dumps(texts, ensure_ascii=False))
        meta = {"model_id": model_id, "pooling": pooling, "prefix": prefix,
                "n_texts": len(texts), "dim": int(emb.shape[1]), "seconds": round(dt, 1),
                "sentences_per_s": round(len(texts) / dt, 1),
                "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                "device": "cpu", "gpu_hours": 0.0}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta))


if __name__ == "__main__":
    main()
