# H-C: when does cross-script invariance help or hurt? (dev split only)

Invariance-only removal (top-k eigenvectors of the pair-difference scatter), 998 pairs per source, 5 resamples, C chosen on in-script dev. Primary k = 16. Macro-F1 x 100.

## Cross-script F1 by pair source (k = 16)

| corpus | encoder | none | human Ar-Az | synth Ar-Az | human Az-CS | human all | in-script none | in-script human all |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pilot_myc | e5-small | 51.8 | 65.2 | 64.2 | 63.0 | 62.9 | 79.2 | 78.3 |
| pilot_narabizi | e5-small | 67.6 | 78.9 | 75.2 | 67.9 | 73.2 | 83.1 | 79.3 |
| pilot_narabizi_topic | e5-small | 29.4 | 35.6 | 29.5 | 29.7 | 32.5 | 53.8 | 49.4 |
| pilot_tsac | e5-small | 72.3 | 70.4 | 67.2 | 75.0 | 75.1 | 90.7 | 88.8 |
| pilot_myc | labse | 68.0 | 69.1 | 66.1 | 71.2 | 69.6 | 80.2 | 79.7 |
| pilot_narabizi | labse | 64.7 | 68.4 | 63.3 | 62.5 | 67.3 | 77.2 | 69.6 |
| pilot_narabizi_topic | labse | 23.1 | 27.7 | 26.9 | 33.1 | 27.2 | 49.5 | 44.6 |
| pilot_tsac | labse | 71.7 | 72.4 | 68.6 | 72.9 | 69.6 | 91.8 | 90.0 |
| pilot_myc | xlmr-base | 60.1 | 64.3 | 58.5 | 67.3 | 63.6 | 74.3 | 72.1 |
| pilot_narabizi | xlmr-base | 47.4 | 48.7 | 57.3 | 47.3 | 48.8 | 65.1 | 65.9 |
| pilot_narabizi_topic | xlmr-base | 12.4 | 20.7 | 20.6 | 21.5 | 20.4 | 21.5 | 23.1 |
| pilot_tsac | xlmr-base | 46.4 | 58.9 | 51.1 | 58.2 | 57.1 | 72.2 | 73.9 |

## Paired comparisons at k = 16 (grouped bootstrap, Holm within each comparison over 12 cells)

### human Ar-Az pairs vs no removal

| cell | delta | 95% CI | p | p_holm |
| --- | ---: | --- | ---: | ---: |
| pilot_myc|e5-small | +13.4 | [+7.9, +18.4] | 0.000 | 0.000 |
| pilot_narabizi|e5-small | +11.3 | [+1.3, +21.7] | 0.024 | 0.192 |
| pilot_narabizi_topic|e5-small | +6.2 | [+1.5, +10.8] | 0.012 | 0.108 |
| pilot_tsac|e5-small | -1.9 | [-5.1, +1.2] | 0.212 | 1.000 |
| pilot_myc|labse | +1.1 | [-3.1, +5.4] | 0.654 | 1.000 |
| pilot_narabizi|labse | +3.8 | [-4.0, +12.4] | 0.384 | 1.000 |
| pilot_narabizi_topic|labse | +4.6 | [-0.1, +9.9] | 0.064 | 0.392 |
| pilot_tsac|labse | +0.7 | [-1.7, +3.0] | 0.584 | 1.000 |
| pilot_myc|xlmr-base | +4.1 | [-0.1, +8.9] | 0.056 | 0.392 |
| pilot_narabizi|xlmr-base | +1.2 | [-8.9, +10.9] | 0.770 | 1.000 |
| pilot_narabizi_topic|xlmr-base | +8.3 | [+3.5, +12.9] | 0.004 | 0.040 |
| pilot_tsac|xlmr-base | +12.6 | [+8.6, +16.4] | 0.000 | 0.000 |

Intervals above 0: 5/12; below 0: 0/12; mean delta +5.45.

### synthetic vs human Ar-Az pairs

| cell | delta | 95% CI | p | p_holm |
| --- | ---: | --- | ---: | ---: |
| pilot_myc|e5-small | -1.0 | [-3.7, +1.6] | 0.460 | 1.000 |
| pilot_narabizi|e5-small | -3.7 | [-13.6, +5.5] | 0.384 | 1.000 |
| pilot_narabizi_topic|e5-small | -6.1 | [-11.0, -1.7] | 0.000 | 0.000 |
| pilot_tsac|e5-small | -3.2 | [-5.4, -1.0] | 0.006 | 0.054 |
| pilot_myc|labse | -3.0 | [-6.6, +0.6] | 0.124 | 0.744 |
| pilot_narabizi|labse | -5.2 | [-15.6, +5.2] | 0.326 | 1.000 |
| pilot_narabizi_topic|labse | -0.8 | [-4.4, +2.7] | 0.630 | 1.000 |
| pilot_tsac|labse | -3.8 | [-6.0, -1.6] | 0.002 | 0.020 |
| pilot_myc|xlmr-base | -5.7 | [-11.1, -1.1] | 0.028 | 0.196 |
| pilot_narabizi|xlmr-base | +8.7 | [+1.7, +15.4] | 0.010 | 0.080 |
| pilot_narabizi_topic|xlmr-base | -0.1 | [-3.0, +2.8] | 0.896 | 1.000 |
| pilot_tsac|xlmr-base | -7.8 | [-11.0, -4.4] | 0.000 | 0.000 |

Intervals above 0: 1/12; below 0: 5/12; mean delta -2.66.

### Az-codeswitch vs Ar-Az pairs

| cell | delta | 95% CI | p | p_holm |
| --- | ---: | --- | ---: | ---: |
| pilot_myc|e5-small | -2.2 | [-5.5, +1.5] | 0.258 | 1.000 |
| pilot_narabizi|e5-small | -11.0 | [-22.2, -1.1] | 0.036 | 0.360 |
| pilot_narabizi_topic|e5-small | -5.9 | [-10.7, -1.1] | 0.008 | 0.088 |
| pilot_tsac|e5-small | +4.6 | [+1.8, +7.5] | 0.002 | 0.024 |
| pilot_myc|labse | +2.1 | [-1.2, +5.1] | 0.218 | 1.000 |
| pilot_narabizi|labse | -5.9 | [-14.3, +2.2] | 0.146 | 1.000 |
| pilot_narabizi_topic|labse | +5.3 | [-1.3, +11.6] | 0.110 | 0.972 |
| pilot_tsac|labse | +0.5 | [-2.2, +3.4] | 0.744 | 1.000 |
| pilot_myc|xlmr-base | +3.0 | [-0.5, +6.3] | 0.108 | 0.972 |
| pilot_narabizi|xlmr-base | -1.4 | [-8.6, +5.2] | 0.672 | 1.000 |
| pilot_narabizi_topic|xlmr-base | +0.8 | [-1.7, +3.7] | 0.594 | 1.000 |
| pilot_tsac|xlmr-base | -0.7 | [-3.5, +2.2] | 0.620 | 1.000 |

Intervals above 0: 1/12; below 0: 2/12; mean delta -0.90.

### all human pairs vs no removal

| cell | delta | 95% CI | p | p_holm |
| --- | ---: | --- | ---: | ---: |
| pilot_myc|e5-small | +11.0 | [+5.5, +16.5] | 0.000 | 0.000 |
| pilot_narabizi|e5-small | +5.5 | [-2.7, +14.4] | 0.184 | 0.860 |
| pilot_narabizi_topic|e5-small | +3.1 | [+0.2, +6.0] | 0.036 | 0.288 |
| pilot_tsac|e5-small | +2.8 | [-0.4, +5.9] | 0.086 | 0.602 |
| pilot_myc|labse | +1.6 | [-2.1, +5.2] | 0.412 | 1.000 |
| pilot_narabizi|labse | +2.6 | [-5.5, +10.8] | 0.544 | 1.000 |
| pilot_narabizi_topic|labse | +4.1 | [+0.5, +8.2] | 0.024 | 0.216 |
| pilot_tsac|labse | -2.0 | [-4.9, +0.7] | 0.142 | 0.852 |
| pilot_myc|xlmr-base | +3.4 | [-1.1, +8.4] | 0.172 | 0.860 |
| pilot_narabizi|xlmr-base | +1.4 | [-8.6, +11.8] | 0.788 | 1.000 |
| pilot_narabizi_topic|xlmr-base | +8.0 | [+3.4, +12.6] | 0.002 | 0.020 |
| pilot_tsac|xlmr-base | +10.8 | [+6.8, +14.6] | 0.000 | 0.000 |

Intervals above 0: 5/12; below 0: 0/12; mean delta +4.36.

## Entanglement and dose-response (human all pairs)

E_k = share of the Arabic-script training task scatter inside the top-k invariance subspace (random expectation k/d: 0.010-0.17).

| cell | k | E_k | raw gap | d cross | d in-script | cross minus derom-encode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pilot_myc|e5-small | 4 | 0.14 | 27.4 | +11.9 | +0.2 | -2.9 |
| pilot_myc|e5-small | 16 | 0.39 | 27.4 | +11.0 | -0.8 | -3.8 |
| pilot_myc|e5-small | 64 | 0.66 | 27.4 | +8.3 | -1.4 | -6.5 |
| pilot_narabizi|e5-small | 4 | 0.05 | 15.5 | +2.8 | -0.3 | +0.3 |
| pilot_narabizi|e5-small | 16 | 0.52 | 15.5 | +5.5 | -3.8 | +3.0 |
| pilot_narabizi|e5-small | 64 | 0.70 | 15.5 | +12.6 | -3.5 | +10.0 |
| pilot_narabizi_topic|e5-small | 4 | 0.07 | 24.4 | +3.1 | -1.7 | +1.1 |
| pilot_narabizi_topic|e5-small | 16 | 0.31 | 24.4 | +3.1 | -4.4 | +1.2 |
| pilot_narabizi_topic|e5-small | 64 | 0.67 | 24.4 | +1.7 | -12.8 | -0.3 |
| pilot_tsac|e5-small | 4 | 0.03 | 18.4 | +3.1 | +0.2 | -4.8 |
| pilot_tsac|e5-small | 16 | 0.48 | 18.4 | +2.8 | -1.9 | -5.1 |
| pilot_tsac|e5-small | 64 | 0.67 | 18.4 | +0.6 | -5.1 | -7.3 |
| pilot_myc|labse | 4 | 0.17 | 12.2 | -0.1 | +0.1 | +9.2 |
| pilot_myc|labse | 16 | 0.45 | 12.2 | +1.6 | -0.5 | +10.8 |
| pilot_myc|labse | 64 | 0.63 | 12.2 | +1.2 | -2.0 | +10.4 |
| pilot_narabizi|labse | 4 | 0.18 | 12.6 | +2.7 | +2.0 | +0.4 |
| pilot_narabizi|labse | 16 | 0.53 | 12.6 | +2.6 | -7.6 | +0.3 |
| pilot_narabizi|labse | 64 | 0.81 | 12.6 | -4.4 | -9.3 | -6.7 |
| pilot_narabizi_topic|labse | 4 | 0.29 | 26.4 | +4.8 | -2.7 | -3.1 |
| pilot_narabizi_topic|labse | 16 | 0.48 | 26.4 | +4.1 | -4.9 | -3.8 |
| pilot_narabizi_topic|labse | 64 | 0.75 | 26.4 | +5.2 | -14.0 | -2.7 |
| pilot_tsac|labse | 4 | 0.11 | 20.2 | -1.4 | -0.1 | -1.1 |
| pilot_tsac|labse | 16 | 0.36 | 20.2 | -2.0 | -1.8 | -1.7 |
| pilot_tsac|labse | 64 | 0.68 | 20.2 | -3.0 | -2.8 | -2.7 |
| pilot_myc|xlmr-base | 4 | 0.64 | 14.2 | +3.9 | +0.3 | +4.4 |
| pilot_myc|xlmr-base | 16 | 0.76 | 14.2 | +3.4 | -2.2 | +3.9 |
| pilot_myc|xlmr-base | 64 | 0.88 | 14.2 | +5.5 | -5.6 | +6.0 |
| pilot_narabizi|xlmr-base | 4 | 0.07 | 17.6 | +4.4 | +2.7 | -4.1 |
| pilot_narabizi|xlmr-base | 16 | 0.35 | 17.6 | +1.4 | +0.9 | -7.2 |
| pilot_narabizi|xlmr-base | 64 | 0.77 | 17.6 | -7.3 | -26.1 | -15.9 |
| pilot_narabizi_topic|xlmr-base | 4 | 0.56 | 9.1 | +7.3 | +0.7 | +1.6 |
| pilot_narabizi_topic|xlmr-base | 16 | 0.76 | 9.1 | +8.0 | +1.6 | +2.2 |
| pilot_narabizi_topic|xlmr-base | 64 | 0.87 | 9.1 | +7.7 | -4.9 | +1.9 |
| pilot_tsac|xlmr-base | 4 | 0.38 | 25.9 | +12.5 | +1.5 | +11.4 |
| pilot_tsac|xlmr-base | 16 | 0.62 | 25.9 | +10.8 | +1.7 | +9.6 |
| pilot_tsac|xlmr-base | 64 | 0.78 | 25.9 | -0.9 | -4.5 | -2.0 |

| k | relation | Spearman rho | p | n |
| --- | --- | ---: | ---: | ---: |
| k=4 | E~d_in | +0.34 | 0.286 | 12 |
| k=4 | E~d_cross | +0.42 | 0.175 | 12 |
| k=4 | raw_gap~d_cross | +0.42 | 0.175 | 12 |
| k=4 | E~d_vs_derom | +0.60 | 0.039 | 12 |
| k=16 | E~d_in | +0.11 | 0.729 | 12 |
| k=16 | E~d_cross | +0.47 | 0.124 | 12 |
| k=16 | raw_gap~d_cross | +0.31 | 0.319 | 12 |
| k=16 | E~d_vs_derom | +0.47 | 0.124 | 12 |
| k=64 | E~d_in | -0.45 | 0.145 | 12 |
| k=64 | E~d_cross | -0.09 | 0.779 | 12 |
| k=64 | raw_gap~d_cross | +0.03 | 0.914 | 12 |
| k=64 | E~d_vs_derom | -0.03 | 0.931 | 12 |
| k=all | E~d_in | -0.57 | 0.000 | 36 |
| k=all | E~d_cross | +0.00 | 0.990 | 36 |
| k=all | raw_gap~d_cross | +0.21 | 0.221 | 36 |
| k=all | E~d_vs_derom | +0.08 | 0.660 | 36 |

Cells share encoders and corpora, so the correlations are descriptive (n = 12 per k, not independent).
