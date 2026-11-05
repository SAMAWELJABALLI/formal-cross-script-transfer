# Troubleshooting

**`pip install -e .` downloads a very large PyTorch wheel.** On Linux the default PyPI wheel bundles CUDA
libraries. Everything here runs on CPU, so install the CPU wheel first:
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `python -m pip install -e ".[test]"`.

**`ModuleNotFoundError: formal_si`.** Install the package (`python -m pip install -e .`) or run with
`PYTHONPATH=src`. The paper scripts in `scripts/paper/` import `formal_si` and the local helper `_paths.py`. Run
them as `python scripts/paper/<name>.py`. They change into the repository root themselves.

**`reproduce_tables.py` reports `NEEDS CORPUS`.** That is expected without the rebuilt corpus. Six count macros and
`data_stats.tex` are computed from `data/examples.jsonl`. Rebuild it as described in `docs/DATASETS.md`. With
the corpus present, all 431 macros reproduce.

**`reproduce_figures.py` reports `SKIPPED` for `figure2_spectral_geometry`.** That figure uses matplotlib's
LaTeX text rendering (packages `libertine`, `libertinust1math`, `amsmath`) and `pdftoppm` (poppler) for the PNG.
Install a TeX distribution and poppler, or use the archived PDF.

**PNG comparisons report small pixel differences.** Font hinting, FreeType and matplotlib versions change
anti-aliasing. The plotted values are compared exactly through the `*_values.json` files. Compare the
figures visually.

**Statistics differ in the last digits.** See "Result of the release verification" in `docs/REPRODUCIBILITY.md`.
Use `--atol 1e-9` for a numerical comparison.

**`RuntimeWarning: Mean of empty slice` while recomputing post-confirmatory statistics.** This is emitted by the
original analysis code when it averages over an empty selection. It does not affect the results. The regenerated files are byte-identical to the archive.

**The pilot re-analysis (`--steps pilot`) is slow.** It bootstraps leave-one-corpus-out model selection over many
settings and did not finish within 30 minutes on a shared 4-core machine. It is therefore not among the default
steps and was not verified for this release. Run it on its own with `python scripts/reproduce_statistics.py --steps pilot`.

**Hugging Face downloads fail or a newer model is loaded.** Level B needs access to `huggingface.co`. The
encoder revisions are pinned in `src/formal_si/encode.py` (`REVISIONS`) and `src/formal_si/encode_labse_st.py`
(`REV`). Do not remove the pins, or the run may use a different model version than the archived results.

**A Level B run overwrote archived files.** Level B writes into `runs/`, `reports/` and `cache*/`. Restore
them with `git checkout -- runs reports` and run Level B in a separate clone.
