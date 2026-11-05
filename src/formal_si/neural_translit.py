"""Post-confirmatory control: character-level neural token transliterator (Arabizi -> Arabic script).

A GRU encoder-decoder with additive attention over characters, trained on token pairs aligned in
human sentence pairs (TArC train + NArabizi train; see translit.aligned_token_pairs). Used as a hybrid:
dictionary lookup for seen tokens, neural decoding for unseen tokens (replacing the rule fallback).
Three models: two cross-fitting folds (for the 998 NArabizi pair sentences) and one full model (for input
canonicalisation). CPU only. Seeds fixed.

Usage: python -m formal_si.neural_translit --out cache_neural_translit
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

from .translit import DictTransliterator, _norm_tok, aligned_token_pairs, fold_of, resources
from .text.normalize import deromanize, normalize_arabic, normalize_arabizi

PAD, BOS, EOS = 0, 1, 2


class Vocab:
    def __init__(self, strings):
        chars = sorted({c for s in strings for c in s})
        self.itos = ["<pad>", "<s>", "</s>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def enc(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]


class Seq2Seq(nn.Module):
    def __init__(self, vs, vt, emb=64, hid=128):
        super().__init__()
        self.se, self.te = nn.Embedding(vs, emb, padding_idx=0), nn.Embedding(vt, emb, padding_idx=0)
        self.enc = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * hid, hid)
        self.dec = nn.GRUCell(emb + 2 * hid, hid)
        self.att_q, self.att_k = nn.Linear(hid, hid, bias=False), nn.Linear(2 * hid, hid, bias=False)
        self.att_v = nn.Linear(hid, 1, bias=False)
        self.out = nn.Linear(hid + 2 * hid, vt)

    def encode(self, src):
        mask = src != PAD
        h, _ = self.enc(self.se(src))
        h0 = torch.tanh(self.bridge((h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)))
        return h, mask, h0

    def step(self, y, s, h, mask, keys):
        e = self.att_v(torch.tanh(self.att_q(s).unsqueeze(1) + keys)).squeeze(-1).masked_fill(~mask, -1e9)
        ctx = (torch.softmax(e, -1).unsqueeze(-1) * h).sum(1)
        s = self.dec(torch.cat([self.te(y), ctx], -1), s)
        return self.out(torch.cat([s, ctx], -1)), s

    def forward(self, src, tgt):
        h, mask, s = self.encode(src)
        keys = self.att_k(h)
        logits = []
        for t in range(tgt.size(1) - 1):
            lo, s = self.step(tgt[:, t], s, h, mask, keys)
            logits.append(lo)
        return torch.stack(logits, 1)

    @torch.no_grad()
    def greedy(self, src, max_len=40):
        h, mask, s = self.encode(src)
        keys = self.att_k(h)
        y = torch.full((src.size(0),), BOS, dtype=torch.long)
        out = []
        for _ in range(max_len):
            lo, s = self.step(y, s, h, mask, keys)
            y = lo.argmax(-1)
            out.append(y)
        return torch.stack(out, 1)


def pad(seqs):
    m = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (m - len(s)) for s in seqs])


class NeuralTransliterator:
    def __init__(self, dictionary: DictTransliterator, model, sv, tv):
        self.d, self.m, self.sv, self.tv, self.cache = dictionary, model, sv, tv, {}

    def _neural(self, toks):
        todo = [t for t in toks if t not in self.cache]
        for i in range(0, len(todo), 256):
            b = todo[i:i + 256]
            src = pad([self.sv.enc(t)[:30] or [3] for t in b])
            out = self.m.greedy(src)
            for t, row in zip(b, out.tolist()):
                chars = []
                for c in row:
                    if c == EOS:
                        break
                    if c > EOS:
                        chars.append(self.tv.itos[c])
                self.cache[t] = "".join(chars) or normalize_arabic(deromanize(t))
        return [self.cache[t] for t in toks]

    def transliterate_many(self, texts):
        keyed = [[_norm_tok(t) for t in normalize_arabizi(x).split()] for x in texts]
        unseen = sorted({k for ks in keyed for k in ks if k and k not in self.d.table and any(c.isalnum() for c in k)})
        self._neural(unseen)
        out = []
        for ks in keyed:
            toks = []
            for k in ks:
                if not k:
                    continue
                toks.append(self.d.table.get(k) or self.cache.get(k) or normalize_arabic(deromanize(k)))
            out.append(" ".join(t for t in toks if t))
        return out


def train_model(pairs, seed=0, epochs=12, threads=2, log=None):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(threads)
    tok_pairs = Counter(aligned_token_pairs(pairs))
    data = [(a, r) for (a, r), _ in tok_pairs.items() if 0 < len(a) <= 30 and 0 < len(r) <= 30]
    sv, tv = Vocab([a for a, _ in data]), Vocab([r for _, r in data])
    model = Seq2Seq(len(sv.itos), len(tv.itos))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss(ignore_index=PAD)
    t0 = time.time()
    for ep in range(epochs):
        random.shuffle(data)
        tot = 0.0
        for i in range(0, len(data), 128):
            b = data[i:i + 128]
            src = pad([sv.enc(a) for a, _ in b])
            tgt = pad([[BOS] + tv.enc(r) + [EOS] for _, r in b])
            lo = model(src, tgt)
            loss = lossf(lo.reshape(-1, lo.size(-1)), tgt[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        if log is not None:
            log.append({"epoch": ep, "loss": tot / len(data), "seconds": round(time.time() - t0, 1)})
    model.eval()
    return model, sv, tv, len(data)


def build(examples, threads=2):
    """Returns (cross_fitted {group: translit}, full NeuralTransliterator, training log)."""
    tarc, nar = resources(examples)
    logs, cf = {}, {}
    for fold in (0, 1):
        tr = tarc + [(fm["arabizi"], fm["arabic"]) for g, fm in nar.items() if fold_of(g) != fold]
        lg = []
        m, sv, tv, n = train_model(tr, seed=fold, threads=threads, log=lg)
        logs[f"fold{fold}"] = {"n_token_pairs": n, "log": lg}
        nt = NeuralTransliterator(DictTransliterator().fit(tr), m, sv, tv)
        gs = [g for g in nar if fold_of(g) == fold]
        for g, t in zip(gs, nt.transliterate_many([nar[g]["arabizi"] for g in gs])):
            cf[g] = t
    full_pairs = tarc + [(fm["arabizi"], fm["arabic"]) for fm in nar.values()]
    lg = []
    m, sv, tv, n = train_model(full_pairs, seed=2, threads=threads, log=lg)
    logs["full"] = {"n_token_pairs": n, "log": lg}
    return cf, NeuralTransliterator(DictTransliterator().fit(full_pairs), m, sv, tv), logs


def main():
    from .confirm import load
    from .translit import cer
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cache_neural_translit")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    ex = load(include_test=True)
    trdev = [e for e in ex if e.split != "test"]
    t0 = time.time()
    cf, full, logs = build(trdev, a.threads)
    # Transliterate every non-Arabic-script row of the four confirmatory tasks (all splits) with the full model.
    rows = [e for e in ex if e.form != "arabic" and e.source in ("narabizi", "tsac", "myc") and e.label]
    uniq = sorted({e.text for e in rows})
    full_map = dict(zip(uniq, full.transliterate_many(uniq)))
    _, nar = resources(trdev)
    dev = {}
    for e in trdev:
        if e.split == "dev" and e.source == "narabizi":
            dev.setdefault(e.group, {})[e.form] = e.text
    dev = [fm for fm in dev.values() if "arabizi" in fm and "arabic" in fm]
    q = {"train_cross_fitted_cer_neural_hybrid": sum(cer(cf[g], normalize_arabic(fm["arabic"])) for g, fm in nar.items()) / len(nar),
         "dev_cer_neural_hybrid_full": sum(cer(full_map.get(fm["arabizi"]) or full.transliterate_many([fm["arabizi"]])[0],
                                               normalize_arabic(fm["arabic"])) for fm in dev) / len(dev),
         "seconds_total": round(time.time() - t0, 1), "threads": a.threads}
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "cross_fitted_pairs.json").write_text(json.dumps(cf, ensure_ascii=False))
    (d / "full_input_map.json").write_text(json.dumps(full_map, ensure_ascii=False))
    (d / "train_log.json").write_text(json.dumps(logs, indent=1))
    (d / "quality.json").write_text(json.dumps(q, indent=2))
    print(json.dumps(q, indent=2))


if __name__ == "__main__":
    main()
