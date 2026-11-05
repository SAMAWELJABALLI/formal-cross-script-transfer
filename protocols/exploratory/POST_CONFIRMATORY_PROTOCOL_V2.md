# Post-confirmatory protocol v2 (exploratory controls requested in manuscript revision)

Written 2026-11-04 ~18:10 UTC, AFTER the confirmatory test evaluation (protocol v1) had been run and inspected,
and BEFORE any of the systems below were evaluated on dev or test. Everything here is therefore
POST-CONFIRMATORY: it cannot change the v1 decisions and is reported separately as exploratory.
Fixed from v1 (unchanged): tasks, splits, emoji stripping, classifier, C selection on in-script dev, 5 resamples
with the v1 seeds, k = 16, 998-pair budget, grouped paired bootstrap (2,000 draws), Holm over cells.

## New systems
- S1T  Tunisian human pairs: 998 TArC TRAIN groups whose code-switched layer is 100% Arabic script
       (2,040 available; fixed random subset, seed 0), paired with their human Arabizi layer.
- S3c  neural-hybrid pairs: the same 998 NArabizi Arabizi sentences, Arabic side from a character GRU
       seq2seq transliterator with dictionary lookup for seen tokens (formal_si/neural_translit.py),
       cross-fitted in the same 2 folds as the dictionary.
- S4c  neural-hybrid input canonicalisation (full model trained on TArC + NArabizi train).
- labse-st  LaBSE with its sentence-transformers head (CLS -> Dense tanh -> L2); all v1 systems re-run.

## Exploratory contrasts (reported with CIs; no confirmatory claims)
- E1 dialect vs corpus: on TSAC (Tunisian target, corpus independent of both pair sources) compare S1T vs S1;
  on NArabizi tasks (Algerian, same corpus as S1 pairs) compare S1 vs S1T.
- E2 better transliterator: S1 vs S3c; S3c vs S3.  E3 input canonicalisation: S4c vs S4b, S1 vs S4c.
- E4 pooling: H1-H3 for labse-st.
Dev results are computed and reported first; test is then run once with the same code.
