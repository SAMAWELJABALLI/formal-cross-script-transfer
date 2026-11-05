# Experiments and their status

Every archived result belongs to one of five analysis families. Their status was fixed when the analysis
was run and is **not** changed by this release. The order below is chronological. All times are UTC, on
2026-11-04, and were recorded by the machine that ran the analysis.

| # | Family | Status | Split(s) | Code | Archived outputs |
| --- | --- | --- | --- | --- | --- |
| 0 | Pilot and H-C analysis | pre-registration exploration, used to choose the fixed elements of the protocol | dev only | `formal_si.run`, `formal_si.analyze`, `formal_si.hc`, `formal_si.hc_analyze`, `formal_si.baselines_text` | `runs/pilot*`, `runs/hc`, `runs/text_baselines*`, `reports/pilot`, `reports/hc` |
| 1 | **Confirmatory study** | **pre-registered** (protocol frozen at 16:32:06, test evaluated once at about 17:14) | dev check, then test once | `formal_si.confirm` | `runs/confirm/{dev,test,curve_dev}` |
| 2 | Dependency-aware re-analysis | post-hoc sensitivity analysis of the frozen predictions (no refitting) | test predictions | `formal_si.depstats` | `reports/depstats/test.json` |
| 3 | Post-confirmatory controls (protocol v2) | **exploratory**, specified after the confirmatory test had been run and inspected | dev, then test | `formal_si.confirm2`, `formal_si.neural_translit`, `formal_si.encode_labse_st` | `runs/confirm2/{dev,test}` |
| 4 | Spectral geometry and error analysis | **descriptive**, no hypothesis tests on test embeddings | dev embeddings; saved test predictions | `formal_si.geometry`, `scripts/paper/v3_geometry.py`, `scripts/paper/v3_prediction_analysis.py` | `reports/geometry`, `reports/v3` |

## 1. Confirmatory study (pre-registered)

- Protocol: [`protocols/confirmatory/CONFIRMATORY_PROTOCOL.md`](../protocols/confirmatory/CONFIRMATORY_PROTOCOL.md).
  SHA-256 `762c3487caca278e1c872515cf236ad0303ad5d4fc4eec92aa67e4bedb98988f`, hash recorded at 16:32:06.
  The protocol was frozen before any test-split string was encoded or read.
- Code at the test run: `protocols/confirmatory/code_at_test_run.sha256`, recorded at 17:14:11.
  `formal_si/confirm.py` hashes to `36b8c779…6c50e2` and `formal_si/translit.py` to `14387a8a…6aff`. Both files are
  unchanged in this release (`src/formal_si/`), and `scripts/verify_artifacts.py` checks the hashes.
- Hypotheses (cross-script macro-F1, train on Arabic script and evaluate on human Arabizi; 4 tasks × 3 encoders = 12 cells):
  H1 S1 > S0, H2 S1 > S2, H3 S1 > S3. A hypothesis is confirmed if the 95% CI of the 12-cell mean excludes 0 in
  the predicted direction.
- Archived test result (`runs/confirm/test/stats.json`): H1 +5.35 [+3.50, +7.31], H2 +4.43 [+2.73, +6.01],
  H3 +1.91 [+0.77, +2.96]. All three are confirmed under the protocol's criterion.
- Secondary comparisons (reported, not confirmatory): S1 vs S4, S1 vs S4b, S4b vs S4, S3 vs S2.
- Learning curve (`runs/confirm/curve_dev`): dev only, as the protocol specifies.
- Bootstrap draws. The protocol fixes 2,000 grouped paired bootstrap draws, and the test analysis uses 2,000.
  The archived **dev** check (`runs/confirm/dev/stats.json`) was computed with **1,000** draws. This was found
  during release verification: the file is reproduced byte for byte with 1,000 draws and not with 2,000. It
  affects only dev CIs and p-values. No confirmatory decision uses dev statistics.
- Deviation already disclosed in the manuscript: a JSON-serialisation fix in the statistics code, made after the
  dev run and before the test run, with no effect on any computation.
- Encoder pooling: the confirmatory LaBSE system uses the **CLS vector of the last hidden state without the
  sentence-transformers dense head**, as the protocol specifies. This is kept as is. The full head is a separate,
  exploratory system (family 3).

## 2. Dependency-aware re-analysis (post-hoc)

`formal_si.depstats` re-analyses the frozen test predictions with a joint grouped bootstrap that respects the
shared test sentences across encoders and across the two NArabizi tasks. It adds task-level and corpus-level
summaries and leave-one-out means. It is a sensitivity analysis and does not replace the protocol's
analysis. It refits nothing.

## 3. Post-confirmatory controls (exploratory)

- Protocol: [`protocols/exploratory/POST_CONFIRMATORY_PROTOCOL_V2.md`](../protocols/exploratory/POST_CONFIRMATORY_PROTOCOL_V2.md).
  SHA-256 `1ca102447bb7f4b13eb5767da0d2b3b844d16a7e8b5e797fb6d1cb7704516824`, hash recorded at 17:57:30. The text
  says it was written at about 18:10. The machine-recorded hash time is the authoritative one; both are after the
  confirmatory test run.
- New systems: S1T (Tunisian TArC pairs), S3c (neural-hybrid transliterator pairs), S4c (neural-hybrid input
  transliteration), S5c (character n-grams on neural-transliterated input), and **LaBSE-ST**. LaBSE-ST is LaBSE
  with its sentence-transformers head (CLS, then Dense with tanh, then L2), revision `836121a`. All v1 systems are
  re-run for it.
- **LaBSE (CLS, confirmatory) and LaBSE-ST (post-confirmatory) are distinct encoders** with distinct names in
  every file: `labse` and `labse-st`. LaBSE-ST results must not be read as confirmatory.
- Contrasts E1–E4 are reported with CIs and no confirmatory claims. E4 repeats H1–H3 for LaBSE-ST and is
  therefore exploratory.
- `runs/confirm2/<split>/predictions.main.jsonl` holds the three original encoders and the character baselines.
  `predictions.labsest.jsonl` holds LaBSE-ST. The original directory also held `predictions.jsonl`, the
  concatenation of the two parts in sorted order, which `formal_si.confirm2 stats` writes. It is a byte-for-byte
  duplicate and is not archived. Recomputing the statistics recreates it.
- Determinism check: confirm2 re-ran S0, S1, S3 and S4b for the three original encoders. The fraction of
  re-run predictions identical to the confirmatory ones is stored in
  `reports/v3/predictions_analysis.json:rerun_identical_fraction`. The archived value is 1.0 for S0 and S1, so
  every checked re-run prediction is identical.

## 4. Spectral geometry and error analysis (descriptive)

- `reports/geometry/*.json` and `reports/v3/geometry_v3.json` hold eigenvalue spectra, principal angles between
  pair subspaces, held-out residual energy, script-probe accuracy and task/script subspace overlap. They are
  computed on train and dev embeddings.
- `reports/v3/viz_coords.json` holds low-dimensional coordinates (PCA, t-SNE, UMAP) of dev embeddings for the
  figures, keyed by sentence-group id.
- `reports/v3/predictions_analysis.json` holds per-class F1, confusion matrices and error categories, computed from
  the saved test predictions. Nothing is refitted.
- These analyses are descriptive. They test no hypothesis and did not influence the confirmatory protocol.

## 0. Pilot and H-C (pre-registration exploration)

These dev-only runs selected the fixed elements of the confirmatory protocol: k = 16, the encoders, emoji
stripping, C selection and the pair budget. `configs/pilot_as_handed_over.yaml` is the pilot configuration as
first written. The other `configs/pilot*.yaml` are the configurations actually run. Negation-sensitivity numbers
in these reports (`AUC`) rely on an **unvalidated** rule-based negation set and support no claim.

## Not done

No fine-tuning, no GPU and no adapter training (the adapter code in `src/formal_si/methods/adapter.py` is tested
but was not used for any reported result). The native-speaker validation of the negation set has not been done.
