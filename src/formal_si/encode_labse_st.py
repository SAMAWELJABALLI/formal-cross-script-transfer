"""Post-confirmatory control: LaBSE with its full sentence-transformers head
(CLS -> Dense(768, tanh) -> L2 normalise), revision 836121a. Encodes every string the confirmatory
systems need (formal_si.confirm.needed_texts for dev and test) plus TArC train pairs.
Usage: python -m formal_si.encode_labse_st --out cache_labse_st/labse-st
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

REV = "836121a0533e5664b21c7aacc5d22951f2b8b25b"


def encode(texts, threads=2):
    """Same pooling as main(); used by confirm2 for strings missing from the labse-st cache."""
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE", revision=REV)
    model = AutoModel.from_pretrained("sentence-transformers/LaBSE", revision=REV).eval()
    sd = load_file(hf_hub_download("sentence-transformers/LaBSE", "2_Dense/model.safetensors", revision=REV))
    dense = torch.nn.Linear(768, 768); dense.load_state_dict({"weight": sd["linear.weight"], "bias": sd["linear.bias"]})
    order = np.argsort([len(t) for t in texts]); out = np.zeros((len(texts), 768), np.float32)
    with torch.inference_mode():
        for s in range(0, len(texts), 64):
            idx = order[s:s + 64]
            b = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=128, return_tensors="pt")
            out[idx] = torch.nn.functional.normalize(torch.tanh(dense(model(**b).last_hidden_state[:, 0])), dim=-1).numpy()
    return out


def main():
    from .confirm import load, needed_texts
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="cache_labse_st/labse-st"); ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    texts = set(needed_texts("dev")) | set(needed_texts("test"))
    texts |= {e.text for e in load(False) if e.source == "tarc" and e.split in ("train", "dev")}
    texts = sorted(t for t in texts if t)
    tok = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE", revision=REV)
    model = AutoModel.from_pretrained("sentence-transformers/LaBSE", revision=REV).eval()
    sd = load_file(hf_hub_download("sentence-transformers/LaBSE", "2_Dense/model.safetensors", revision=REV))
    dense = torch.nn.Linear(768, 768); dense.load_state_dict({"weight": sd["linear.weight"], "bias": sd["linear.bias"]})
    order = np.argsort([len(t) for t in texts]); out = np.zeros((len(texts), 768), np.float32); t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(texts), 64):
            idx = order[s:s + 64]
            b = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=128, return_tensors="pt")
            cls = model(**b).last_hidden_state[:, 0]
            out[idx] = torch.nn.functional.normalize(torch.tanh(dense(cls)), dim=-1).numpy()
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    np.save(d / "emb.npy", out.astype(np.float16)); (d / "texts.json").write_text(json.dumps(texts, ensure_ascii=False))
    (d / "meta.json").write_text(json.dumps({"model_id": "sentence-transformers/LaBSE", "revision": REV, "pooling": "cls+dense(tanh)+l2",
                                             "n_texts": len(texts), "seconds": round(time.time() - t0, 1), "threads": a.threads}))
    print("done", len(texts), round(time.time() - t0))


if __name__ == "__main__":
    main()
