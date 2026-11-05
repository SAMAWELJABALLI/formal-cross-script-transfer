# Reproducibility

Two levels are supported.

| Level | What it does | Needs | Time (4 CPU cores) |
| --- | --- | --- | --- |
| **A: from archived outputs** | recomputes the statistics, tables, number macros and figures from the archived predictions and reports, then compares them with the archive | this repository and Python only. No corpus, no model download, no network | about 25 min for the default statistics steps, under 1 min for tables, about 30 s for figures |
| **B: full re-run** | rebuilds the corpus, encodes it with the frozen encoders, retrains the transliterators and re-runs every evaluation | network access to GitHub and Hugging Face, about 2 GB of model downloads, about 1 GB of disk for caches | about 5 CPU-hours for the listed steps (see `docs/COMPUTE.md`) |

The repository is **not self-contained** for Level B. Level B depends on five external dataset repositories and
three Hugging Face models, which this repository pins but does not ship.

## Level A

```bash
python -m pip install -e ".[test]"
python scripts/verify_artifacts.py        # checksums, protocol and code hashes, label-only predictions
python scripts/reproduce_statistics.py    # stats from archived predictions, compared with the archive
python scripts/reproduce_tables.py        # LaTeX tables and \newcommand macros, compared with tables/
python scripts/reproduce_figures.py       # figures, compared with figures/
```

Outputs go to `out/`. You can change this with `FORMAL_OUT=/some/dir`. Archived files are never overwritten.

### Result of the release verification (2026-11-05, fresh virtual environment, Python 3.13)

| Check | Result |
| --- | --- |
| `verify_artifacts.py` | all manifest hashes, both protocol hashes, both code hashes match. Every prediction record holds label strings only |
| Confirmatory statistics, test (`runs/confirm/test/stats.json`, 2,000 draws) | byte-identical |
| Confirmatory statistics, dev (`runs/confirm/dev/stats.json`, 1,000 draws, see `docs/EXPERIMENTS.md`) | byte-identical |
| Dependency-aware re-analysis (`reports/depstats/test.json`) | byte-identical |
| Post-confirmatory statistics, dev and test (`runs/confirm2/*/stats.json`) | byte-identical |
| Prediction analysis (`reports/v3/predictions_analysis.json`) | byte-identical |
| H-C analysis (`reports/hc/hc_analysis.json`) | byte-identical |
| Pilot analysis (`reports/pilot/pilot_analysis.json`, optional step `--steps pilot`) | **not verified**: the run was stopped after 30 minutes on a shared 4-core machine without finishing. It is excluded from the default steps |
| Tables | all generated tables byte-identical. 425 of 431 macros in `numbers.tex` and 199 of 199 in `v3_numbers.tex` identical. The remaining 6 macros and `data_stats.tex` need the corpus and are also identical once it is rebuilt |
| Figures | all 17 manuscript figures regenerated. Values files (`fig_H_curve_values.json`, `fig_J_values.json`, `PROVENANCE.json`, `axes_stats.json`) are byte-identical, and all 11 archived PNGs are pixel-identical. PDF bytes are not compared because they embed a creation date |

Byte identity was observed with the versions in `requirements-lock.txt`. Bootstrap resampling uses NumPy's
`default_rng` (PCG64) with fixed seeds and is stable across NumPy versions, but floating-point summation order
can differ across BLAS libraries, NumPy versions and CPUs. Other environments may therefore produce differences in
the last digits. `reproduce_statistics.py --atol 1e-9` compares numerically instead of exactly.

## Level B

```bash
python -m pip install -e ".[encode,umap,test]"
python -m formal_si.data.fetch && python -m formal_si.data.build    # corpus; verify SHA-256, see docs/DATASETS.md
python -m formal_si.data.audit                                      # reports/data_audit.json
# pilot and H-C (dev only)
python -m formal_si.encode --config configs/pilot.yaml --encoder e5-small labse xlmr-base
python -m formal_si.run --config configs/pilot_narabizi.yaml   # also pilot_tsac, pilot_myc, pilot_narabizi_topic, pilot
python -m formal_si.analyze --runs runs/pilot_narabizi runs/pilot_tsac runs/pilot_myc runs/pilot_narabizi_topic \
    --extra-runs runs/pilot --out reports/pilot/pilot_analysis.md
for e in e5-small labse xlmr-base; do python -m formal_si.hc --encoder $e; done
python -m formal_si.hc_analyze
# confirmatory study (protocols/confirmatory)
python -m formal_si.translit                                         # reports/confirm/translit_quality.json
python -m formal_si.confirm texts && python -m formal_si.confirm encode
python -m formal_si.confirm evaluate --split dev && python -m formal_si.confirm stats --split dev --n-boot 1000
python -m formal_si.confirm curve
python -m formal_si.confirm evaluate --split test && python -m formal_si.confirm stats --split test
python -m formal_si.depstats
# post-confirmatory controls (protocols/exploratory)
python -m formal_si.neural_translit
python -m formal_si.encode_labse_st
python -m formal_si.confirm2 encode --encoder e5-small labse xlmr-base labse-st
for s in dev test; do
  python -m formal_si.confirm2 evaluate --split $s --part main --encoder e5-small labse xlmr-base
  python -m formal_si.confirm2 evaluate --split $s --part labsest --encoder labse-st --no-char
  python -m formal_si.confirm2 stats --split $s
done
# geometry (descriptive)
python -m formal_si.geometry
python scripts/paper/v3_geometry.py
python scripts/paper/v3_prediction_analysis.py
python scripts/paper/make_examples.py
```

A Level B re-run writes into `runs/`, `reports/` and `cache*/`, which **overwrites the archived files** in a working
copy. Run it in a separate clone and compare it with the archive, for example with `git diff --stat`.
Each module's `--help` lists its options. Check the module docstrings for the exact arguments before a long run.

### What to expect from Level B

- **Corpus.** Byte-identical rebuild verified (SHA-256 in `artifacts/corpus_fingerprint.json`).
- **Encoders.** The original runs loaded each model's Hub `main` branch, which resolved to the revisions recorded
  in the protocol. This release pins those revisions in code (`src/formal_si/encode.py`, `REVISIONS`, and
  `src/formal_si/encode_labse_st.py`, `REV`). A re-run therefore cannot silently use a newer model.
  Embeddings are computed on CPU in float32 and cached in float16. Across PyTorch versions, CPU instruction sets
  and thread counts, embeddings can differ in the last bits, so predictions near a decision boundary can flip.
  **Exact bitwise reproduction of Level B is not claimed.** Expect small differences in macro-F1 and CIs.
- **Neural transliterator.** The GRU is trained on CPU with fixed seeds. PyTorch does not guarantee bitwise-equal
  training across versions or thread counts, so S3c, S4c and S5c can differ slightly.
- **Test split.** The protocol allowed a single test evaluation. That evaluation is archived in `runs/confirm/test`.
  A re-run is a replication, not a second confirmatory test.

## Software environment

- `pyproject.toml` gives minimum versions plus the exact `transformers==4.49.0` pin used for encoding.
- `requirements.txt` repeats the direct dependencies.
- `requirements-lock.txt` is the complete `pip freeze` of the fresh environment in which Level A was verified for
  this release. The exact package versions of the original 2026-11-04 runs were not recorded, apart from
  `transformers==4.49.0` and the model revisions. The lock file documents the verification environment, not the
  original one.
- CPU only. No GPU was used for any result.
