"""Low-rank residual adapter on cached embeddings (Section 4, adapter variant).

h' = h + B A h            (ungated)
h' = h + gamma_t * B A h  (task-gated, H-B ablation; gamma_t is a learned d-vector per task)

Loss = CE(y, V h') + alpha * mean ||h'_a - h'_b||^2 over meaning-preserving pairs
       + gamma * mean relu(||h_x - h_n|| - ||h'_x - h'_n||) over meaning-changing pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class LowRankAdapter(nn.Module):
    def __init__(self, dim: int, n_classes: int, rank: int = 16, n_tasks: int = 0):
        super().__init__()
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.B.weight)  # start as the identity map
        self.gates = nn.Parameter(torch.ones(n_tasks, dim)) if n_tasks else None
        self.head = nn.Linear(dim, n_classes)

    def adapt(self, h: torch.Tensor, task: int | None = None) -> torch.Tensor:
        delta = self.B(self.A(h))
        if self.gates is not None:
            if task is None:
                raise ValueError("gated adapter needs a task index")
            delta = delta * self.gates[task]
        return h + delta

    def forward(self, h, task=None):
        return self.head(self.adapt(h, task))

    def n_adapter_params(self) -> int:
        n = self.A.weight.numel() + self.B.weight.numel()
        return n + (self.gates.numel() if self.gates is not None else 0)


@dataclass
class AdapterTrainer:
    rank: int = 16
    alpha: float = 1.0
    gamma: float = 1.0
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 256
    weight_decay: float = 1e-4
    seed: int = 0
    gated: bool = False

    def fit(self, x, y, pair_a=None, pair_b=None, con_a=None, con_b=None, task: int = 0,
            n_tasks: int = 1):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        classes, y_idx = np.unique(np.asarray(y), return_inverse=True)
        self.classes_ = classes
        y_t = torch.as_tensor(y_idx, dtype=torch.long)
        self.task = task if self.gated else None
        self.model = LowRankAdapter(x.shape[1], len(classes), self.rank,
                                    n_tasks if self.gated else 0)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        pa = torch.as_tensor(np.asarray(pair_a), dtype=torch.float32) if pair_a is not None else None
        pb = torch.as_tensor(np.asarray(pair_b), dtype=torch.float32) if pair_b is not None else None
        ca = torch.as_tensor(np.asarray(con_a), dtype=torch.float32) if con_a is not None else None
        cb = torch.as_tensor(np.asarray(con_b), dtype=torch.float32) if con_b is not None else None
        self.history = []
        n = len(x)
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            total = 0.0
            for s in range(0, n, self.batch_size):
                idx = torch.as_tensor(perm[s:s + self.batch_size])
                loss = nn.functional.cross_entropy(self.model(x[idx], self.task), y_t[idx])
                if pa is not None and self.alpha > 0 and len(pa):
                    j = torch.as_tensor(rng.integers(0, len(pa), min(self.batch_size, len(pa))))
                    diff = self.model.adapt(pa[j], self.task) - self.model.adapt(pb[j], self.task)
                    loss = loss + self.alpha * diff.pow(2).sum(1).mean()
                if ca is not None and self.gamma > 0 and len(ca):
                    j = torch.as_tensor(rng.integers(0, len(ca), min(self.batch_size, len(ca))))
                    before = (ca[j] - cb[j]).norm(dim=1)
                    after = (self.model.adapt(ca[j], self.task)
                             - self.model.adapt(cb[j], self.task)).norm(dim=1)
                    loss = loss + self.gamma * torch.relu(before - after).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
            self.history.append(total / n)
        return self

    @torch.no_grad()
    def transform(self, x):
        h = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return self.model.adapt(h, self.task).numpy().astype(np.float64)

    @torch.no_grad()
    def predict(self, x):
        h = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return self.classes_[self.model(h, self.task).argmax(1).numpy()]
