# FORMAL: pair provenance and cross-script transfer in Maghrebi Arabic

Code, protocols and archived results for the manuscript

> **Pair Provenance Shapes Spectral Subspace Geometry for Cross-Script Transfer in Maghrebi Arabic**
> Samawel JABALLI. Manuscript prepared for submission to *ACM Transactions on Asian and Low-Resource Language
> Information Processing (TALLIP)*. It has not been peer reviewed, accepted or published.

## 1. Overview

Multilingual sentence encoders place Arabic-script and Latin-script (Arabizi) renderings of Maghrebi Arabic in
partly separate regions. A classifier trained on Arabic script therefore loses accuracy on Arabizi. A
lightweight correction removes the top-k eigenvectors of the scatter of paired embedding differences:
P_k = I − U_k U_kᵀ, with k = 16. This repository studies **where the pairs come from**. It compares human Arabic-script
renderings of 998 Algerian Arabizi sentences with rule-based, dictionary-based and neural transliterations of the
same sentences, on three frozen encoders and four classification tasks. The design is a pre-registered protocol
with a single test evaluation.

## 2. Main findings (from the archived files)

| Confirmatory hypothesis (test split, 12-cell mean of cross-script macro-F1 differences) | Δ [95% CI] | Verdict under the protocol |
| --- | --- | --- |
| H1 human pairs > no removal | +5.35 [+3.50, +7.31] | confirmed |
| H2 human pairs > rule-transliteration pairs | +4.43 [+2.73, +6.01] | confirmed |
| H3 human pairs > dictionary-transliteration pairs | +1.91 [+0.77, +2.96] | confirmed |

A post-hoc, dependency-aware re-analysis places most of the gain in the corpus that supplied the pairs.
Transliterating the input with a dictionary or neural transliterator matches or exceeds every projection.
Details, secondary comparisons and the exploratory controls are in [`docs/RESULTS.md`](docs/RESULTS.md).

## 3. Status of each analysis

| Analysis | Status | Where |
| --- | --- | --- |
| Pilot and H-C (dev only) | pre-registration exploration | `runs/pilot*`, `runs/hc`, `reports/pilot`, `reports/hc` |
| Confirmatory study H1–H3 | **pre-registered**, single test evaluation | `protocols/confirmatory`, `runs/confirm` |
| Dependency-aware re-analysis | post-hoc sensitivity analysis | `reports/depstats` |
| Post-confirmatory controls (TArC pairs, neural transliterator, LaBSE-ST) | **exploratory**, specified after the test run | `protocols/exploratory`, `runs/confirm2` |
| Spectral geometry and error analysis | **descriptive** | `reports/geometry`, `reports/v3` |

The confirmatory LaBSE system uses the CLS vector **without** the sentence-transformers head, as pre-registered.
LaBSE with its head (`labse-st`) is a separate, exploratory system. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## 4. Repository structure

```
src/formal_si/          research code (package `formal_si`)
  confirm.py            confirmatory study: texts, encode, evaluate, curve, stats  (hash-verified)
  confirm2.py           post-confirmatory controls (exploratory)
  depstats.py           dependency-aware joint bootstrap (post-hoc)
  translit.py           dictionary transliterator, cross-fitting, CER             (hash-verified)
  neural_translit.py    character GRU transliterator (exploratory)
  encode.py, encode_labse_st.py   frozen encoders (pinned revisions)
  geometry.py           spectra, principal angles, overlap (descriptive)
  run.py, analyze.py, hc.py, hc_analyze.py, decide.py, diagnostics.py, views.py, baselines_text.py   pilot / H-C
  data/  eval/  methods/  text/   corpus fetch and build, metrics and bootstrap, projections, normalisation
configs/                pilot configurations (as run, plus the first draft)
protocols/              confirmatory and exploratory protocols with SHA-256 and code-hash records
runs/                   archived predictions (label lists only), gold labels, statistics, timings
reports/                archived analyses: depstats, geometry, v3, pilot, H-C, data audit, transliteration quality
artifacts/              encoding and transliterator metadata, corpus fingerprint, MANIFEST.sha256
tables/                 LaTeX tables and number macros of the manuscript
figures/main/           figures of the main text;  figures/supplementary/  appendix figures
scripts/                verify_artifacts.py, reproduce_{statistics,tables,figures}.py
scripts/paper/          the scripts that generate every table, macro and figure of the manuscript
tests/                  unit and integration tests (synthetic data only)
docs/                   DATASETS, EXPERIMENTS, REPRODUCIBILITY, RESULTS, COMPUTE, TROUBLESHOOTING
```

## 5. Installation

Requires Python ≥ 3.10. Everything runs on CPU.

```bash
git clone https://github.com/SAMAWELJABALLI/formal-cross-script-transfer.git
cd formal-cross-script-transfer
python -m venv .venv && source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: avoids the CUDA wheel
python -m pip install -e ".[test]"                                              # Level A
python -m pip install -e ".[encode,umap,test]"                                  # Level B
```

`requirements.txt` lists the direct dependencies. `requirements-lock.txt` is the exact environment in which this
release was verified.

## 6. Quick verification

```bash
python -m pytest                       # unit and integration tests on synthetic data
python scripts/verify_artifacts.py     # SHA-256 manifest, protocol and code hashes, label-only predictions
```

## 7. Reproducibility levels

- **Level A: from archived predictions** (no corpus, no models, no network):
  `scripts/reproduce_statistics.py`, `scripts/reproduce_tables.py`, `scripts/reproduce_figures.py`.
  All confirmatory, post-hoc, post-confirmatory, H-C and v3 statistics and all generated tables were reproduced
  **byte for byte** during release verification. The slow pilot re-analysis was not verified.
- **Level B: full re-run.** Rebuild the corpus, re-encode, retrain the transliterators and re-evaluate. This depends
  on external datasets and Hugging Face models, so **this repository is not self-contained for Level B**.
  Bitwise identity is not expected for Level B, because of CPU/BLAS and PyTorch nondeterminism.

Commands, expected outputs and caveats: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## 8. Data

No corpus text is included. Five public corpora (NArabizi, TArC, TSAC, TUNIZI, MYC) are downloaded at pinned
commits and built locally. The rebuild was verified to be byte-identical to the corpus used for the results.
Licences differ, and some corpora have none. See [`docs/DATASETS.md`](docs/DATASETS.md) before using the data.

## 9. Models

| Name in files | Hugging Face model | Revision | Pooling |
| --- | --- | --- | --- |
| `e5-small` | `intfloat/multilingual-e5-small` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | mean, prefix `"query: "` |
| `labse` | `sentence-transformers/LaBSE` | `836121a0533e5664b21c7aacc5d22951f2b8b25b` | CLS of last hidden state, no head (confirmatory) |
| `xlmr-base` | `FacebookAI/xlm-roberta-base` | `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` | mean |
| `labse-st` | `sentence-transformers/LaBSE` | `836121a0533e5664b21c7aacc5d22951f2b8b25b` | CLS, then Dense with tanh, then L2 (exploratory) |

All encoders are frozen, use at most 128 tokens and produce L2-normalised embeddings cached in float16. No weights are redistributed.

## 10. Method summary

Logistic regression with C ∈ {0.1, 1, 10}, selected on in-script dev macro-F1. Five 90% training resamples,
with seeds from `numpy.random.default_rng(0)`. Projection with k = 16 from 998 pairs. Statistics use a grouped
paired bootstrap with 2,000 draws on test, Holm correction over the 12 cells, and the 12-cell mean as the
confirmatory estimand. The full specification is in [`protocols/confirmatory/CONFIRMATORY_PROTOCOL.md`](protocols/confirmatory/CONFIRMATORY_PROTOCOL.md).

## 11. Archived results and integrity

`artifacts/MANIFEST.sha256` lists the SHA-256 of every code, protocol, result, table and figure file.
The protocol hashes, recorded before the test evaluation, and the code hashes, recorded at the test evaluation, are in
`protocols/`. The two code files they cover are unchanged in this release. See [`protocols/README.md`](protocols/README.md).

## 12. Compute

All results used CPU only, with 0 GPU-hours. The recorded steps sum to 5.0 wall-clock hours on 4 cores. See [`docs/COMPUTE.md`](docs/COMPUTE.md).

## 13. Limitations

- The pairs come from one Algerian corpus. The gains concentrate in that corpus (see the dependency-aware re-analysis).
- The negation-sensitivity set is rule-generated and has not been validated by native speakers. It supports no claim.
- Level B depends on third-party repositories and models that may change or disappear. Commits and revisions are
  pinned, and the corpus fingerprint detects changes.
- The package versions of the original runs were not recorded beyond `transformers==4.49.0` and the model
  revisions. The lock file documents the verification environment.
- Protocol timestamps come from the system clock and are not third-party attested.
- Corpus-dependent tables (example sentences, data statistics) are regenerated only with the rebuilt corpus.

## 14. Citation

See [`CITATION.cff`](CITATION.cff). No DOI exists yet. Please cite the software and, once available, the article.

## 15. Licence

- Code (`src/`, `scripts/`, `tests/`, `configs/`): [MIT](LICENSE).
- Original archived predictions, statistical results, tables, figures and protocols: [CC BY 4.0](LICENSE-DATA.md),
  subject to the rights of all contributors.
- Third-party datasets and pretrained models are **not** relicensed. Their own terms apply ([`docs/DATASETS.md`](docs/DATASETS.md)).

## 16. Contact and contributions

Please use GitHub issues for questions about the code or the archived results. The archived results and protocols are
frozen records and will not be modified. Corrections will be published as new, clearly labelled files or releases.
