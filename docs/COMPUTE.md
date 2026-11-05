# Compute

All results were produced on CPU (a 4-core cloud container). **No GPU was used: 0 GPU-hours.**

## Recorded wall-clock times

Generated from `artifacts/encoding_meta/**/meta.json`, `artifacts/neural_translit/quality.json` and
`runs/**/timing*.json` (identical to `tables/compute_rows.tex`). "n.r." means the thread count was not recorded.

| Step | Strings | Wall s | Threads |
| --- | ---: | ---: | --- |
| encode pilot strings, mE5-small | 94,014 | 609 | n.r. |
| encode pilot strings, LaBSE | 94,014 | 1,892 | n.r. |
| encode pilot strings, XLM-R | 94,014 | 2,760 | n.r. |
| encode confirmatory strings, mE5-small | 19,446 | 229 | 4 |
| encode confirmatory strings, LaBSE | 19,446 | 559 | 4 |
| encode confirmatory strings, XLM-R | 19,446 | 492 | 4 |
| encode post-conf. strings, mE5-small | 13,402 | 111 | 2 |
| encode post-conf. strings, LaBSE | 13,402 | 292 | 2 |
| encode post-conf. strings, XLM-R | 13,402 | 280 | 2 |
| encode all strings, LaBSE-ST | 83,645 | 3,084 | 2 |
| encode post-conf. strings, LaBSE-ST | 13,405 | 310 | 2 |
| train 3 neural transliterators | | 350 | 1 |
| confirmatory evaluation, dev | | 400 | n.r. |
| confirmatory evaluation, test | | 747 | n.r. |
| post-conf. evaluation, dev (LaBSE-ST part) | | 908 | n.r. |
| post-conf. evaluation, dev (main part) | | 2,059 | n.r. |
| post-conf. evaluation, test (LaBSE-ST part) | | 897 | n.r. |
| post-conf. evaluation, test (main part) | | 2,008 | n.r. |
| **sum of listed steps** | | **17,985 (5.0 h)** | |

Unlisted steps were not timed individually: pilot model selection, H-C analysis, learning curve, geometry,
bootstrap statistics and the transliteration-quality report. The manuscript estimates about 15 CPU core-hours
for the whole project before revision. That figure is an estimate, not a measurement.

## Level A cost (measured during release verification, 4 cores)

| Script | Wall time |
| --- | --- |
| `scripts/verify_artifacts.py` | a few seconds |
| `scripts/reproduce_statistics.py` (default steps) | about 25 min: confirmatory about 5, depstats about 2, post-confirmatory about 7, H-C about 10, v3 a few seconds. The optional pilot step takes over an hour (see `docs/TROUBLESHOOTING.md`) |
| `scripts/reproduce_tables.py` | about 5 s |
| `scripts/reproduce_figures.py` | about 30 s |

Peak memory for Level A is under 2 GB. Level B encoding peaked at about 2.1 GB RSS per encoder
(`artifacts/encoding_meta/cache/*/meta.json`).
