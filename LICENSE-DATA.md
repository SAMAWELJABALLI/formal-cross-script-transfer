# Licence for archived results, tables and figures

Copyright (c) 2026 Samawel JABALLI, subject to the rights of all contributors.

The original research outputs in this repository are licensed under the
**Creative Commons Attribution 4.0 International licence (CC BY 4.0)**:
<https://creativecommons.org/licenses/by/4.0/legalcode>

This covers:

- archived predictions and gold label lists (`runs/`),
- statistical and analysis outputs (`reports/`, `artifacts/`),
- generated LaTeX tables (`tables/`),
- figures (`figures/`),
- the protocol documents (`protocols/`).

The source code (`src/`, `scripts/`, `tests/`, `configs/`) is licensed under the MIT licence (see `LICENSE`).

## What this licence does not cover

- **Third-party datasets.** No corpus text is redistributed here. The corpora (NArabizi, TArC, TSAC,
  TUNIZI, MYC) stay under their own terms, which are summarised as found in `docs/DATASETS.md`. Nothing in
  this repository relicenses them. The gold label lists in `runs/*/gold.json` are derived from those corpora.
  They contain only label strings and sentence-group identifiers, with no text, so that
  the archived predictions can be scored. Users must check the original dataset terms before any use beyond
  verifying the results reported here.
- **Pretrained encoders** (`intfloat/multilingual-e5-small`, `sentence-transformers/LaBSE`,
  `FacebookAI/xlm-roberta-base`). They are not redistributed and stay under the licences published on their
  Hugging Face model cards.
- **Python dependencies.** They stay under their own licences.
