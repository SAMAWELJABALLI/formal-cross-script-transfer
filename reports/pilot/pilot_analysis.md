# FORMAL pilot: per-corpus analysis (dev split only)

Macro-F1 x 100, mean over 5 training resamples. Cross = train Arabic script, evaluate human Arabizi. AUC = negation sensitivity (share of anchors whose rule negation is farther than their human other-script paraphrase; higher = more sensitive). Inv = invariance score on dev human pairs.

## Selection rule: loco

### pilot_myc / e5-small

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 79.2 | 51.8 |  | 0.171 | 0.479 |  | 0.66 |
| derom_encode | - | 78.8 | 66.6 |  | 0.259 | 0.722 |  | 0.68 |
| centering | - | 79.0 | 61.2 |  | 0.176 | 0.523 |  | 0.86 |
| lsar | {'r': 1} | 78.8 | 51.6 |  | 0.172 | 0.483 |  | 0.76 |
| leace | - | 79.2 | 63.6 |  | 0.185 | 0.537 |  | 1.04 |
| splince | - | 79.2 | 51.9 |  | 0.171 | 0.479 |  | 1.23 |
| random | {'k': 4} | 79.2 | 51.6 |  | 0.171 | 0.479 |  | 0.64 |
| formal_si_inv_only | {'k': 16, 'lam0': 0.0, 'beta': 1.0} | 77.6 | 63.8 |  | 0.237 | 0.627 |  | 1.16 |
| formal_si_sel_task | {'k': 64, 'lam0': 0.5, 'beta': 0.0} | 76.9 | 50.1 |  | 0.248 | 0.636 |  | 1.01 |
| formal_si_sel_contrast | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 79.3 | 66.1 |  | 0.273 | 0.654 |  | 1.09 |
| derom+leace | - | 77.9 | 64.0 |  | 0.262 | 0.729 |  | 1.56 |
| derom+formal_si | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 78.0 | 67.3 |  | 0.312 | 0.775 |  | 1.11 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_sel_contrast vs best baseline (derom_encode) | -0.5 | [-5.6, +4.7] | 0.818 | +0.014 |
| selective (best lam0>0) vs invariance-only | +2.3 | [-1.3, +6.0] | 0.232 | +0.036 |
| invariance-only vs derom_encode | -2.8 | [-8.5, +2.3] | 0.304 | -0.022 |
| derom+formal_si vs derom_encode | +0.7 | [-1.9, +3.3] | 0.658 | +0.054 |
| invariance-only vs random removal (k=16) | +11.3 | [+6.4, +16.3] | 0.000 | |

### pilot_myc / labse

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 80.2 | 68.0 |  | 0.203 | 0.618 |  | 1.25 |
| derom_encode | - | 79.8 | 58.8 |  | 0.276 | 0.751 |  | 1.34 |
| centering | - | 80.0 | 63.5 |  | 0.207 | 0.634 |  | 1.94 |
| lsar | {'r': 1} | 80.2 | 68.1 |  | 0.202 | 0.615 |  | 1.41 |
| leace | - | 80.2 | 65.5 |  | 0.216 | 0.636 |  | 2.39 |
| splince | - | 80.3 | 67.6 |  | 0.203 | 0.618 |  | 3.17 |
| random | {'k': 64} | 80.4 | 68.0 |  | 0.204 | 0.618 |  | 1.69 |
| formal_si_inv_only | {'k': 16, 'lam0': 0.0, 'beta': 1.0} | 79.4 | 69.7 |  | 0.250 | 0.691 |  | 2.53 |
| formal_si_sel_task | {'k': 64, 'lam0': 0.5, 'beta': 0.0} | 77.3 | 62.7 |  | 0.243 | 0.701 |  | 2.90 |
| formal_si_sel_contrast | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 78.5 | 67.1 |  | 0.320 | 0.737 |  | 2.16 |
| derom+leace | - | 79.1 | 60.4 |  | 0.277 | 0.753 |  | 3.22 |
| derom+formal_si | {'k': 16, 'lam0': 0.0, 'beta': 1.0} | 78.6 | 67.7 |  | 0.269 | 0.785 |  | 2.09 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_inv_only vs best baseline (lsar) | +1.6 | [-2.4, +5.5] | 0.438 | +0.048 |
| selective (best lam0>0) vs invariance-only | -2.6 | [-6.6, +1.7] | 0.248 | +0.070 |
| invariance-only vs derom_encode | +11.0 | [+5.0, +17.2] | 0.000 | -0.026 |
| derom+formal_si vs derom_encode | +8.9 | [+3.7, +14.2] | 0.000 | -0.007 |
| invariance-only vs random removal (k=16) | +2.1 | [-2.2, +6.2] | 0.356 | |

### pilot_myc / xlmr-base

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 74.3 | 60.1 |  | 0.159 | 0.564 |  | 2.13 |
| derom_encode | - | 74.0 | 59.7 |  | 0.285 | 0.813 |  | 1.41 |
| centering | - | 74.3 | 62.5 |  | 0.201 | 0.642 |  | 2.74 |
| lsar | {'r': 1} | 75.0 | 56.9 |  | 0.133 | 0.325 |  | 1.93 |
| leace | - | 74.4 | 61.5 |  | 0.193 | 0.637 |  | 3.23 |
| splince | - | 74.3 | 61.8 |  | 0.159 | 0.564 |  | 3.61 |
| random | {'k': 4} | 74.5 | 60.5 |  | 0.159 | 0.564 |  | 2.05 |
| formal_si_inv_only | {'k': 4, 'lam0': 0.0, 'beta': 0.0} | 74.8 | 65.4 |  | 0.193 | 0.541 |  | 2.20 |
| formal_si_sel_task | {'k': 4, 'lam0': 0.5, 'beta': 0.0} | 71.6 | 59.9 |  | 0.194 | 0.681 |  | 2.64 |
| formal_si_sel_contrast | {'k': 4, 'lam0': 0.5, 'beta': 1.0} | 74.4 | 65.7 |  | 0.217 | 0.730 |  | 3.18 |
| derom+leace | - | 73.5 | 64.6 |  | 0.293 | 0.789 |  | 3.01 |
| derom+formal_si | {'k': 4, 'lam0': 2.0, 'beta': 1.0} | 73.6 | 64.6 |  | 0.310 | 0.837 |  | 2.12 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_sel_contrast vs best baseline (centering) | +3.2 | [-2.4, +8.5] | 0.258 | +0.016 |
| selective (best lam0>0) vs invariance-only | +0.3 | [-2.6, +3.3] | 0.878 | +0.025 |
| invariance-only vs derom_encode | +5.7 | [+0.1, +11.4] | 0.048 | -0.092 |
| derom+formal_si vs derom_encode | +4.9 | [+1.3, +9.1] | 0.002 | +0.025 |
| invariance-only vs random removal (k=4) | +4.9 | [+1.5, +8.3] | 0.000 | |

### pilot_narabizi / e5-small

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 83.1 | 67.6 | 84.1 | 0.171 | 0.479 |  | 0.16 |
| derom_encode | - | 81.1 | 70.2 | 77.1 | 0.259 | 0.722 |  | 0.15 |
| centering | - | 83.2 | 63.4 | 83.4 | 0.179 | 0.524 |  | 0.19 |
| lsar | {'r': 1} | 82.4 | 64.3 | 84.4 | 0.186 | 0.543 |  | 0.21 |
| leace | - | 82.4 | 67.9 | 83.0 | 0.184 | 0.535 |  | 0.32 |
| splince | - | 82.9 | 68.9 | 83.8 | 0.171 | 0.479 |  | 0.32 |
| random | {'k': 16} | 81.8 | 67.3 | 83.2 | 0.171 | 0.478 |  | 0.18 |
| formal_si_inv_only | {'k': 16, 'lam0': 0.0, 'beta': 1.0} | 80.6 | 74.4 | 79.9 | 0.237 | 0.627 |  | 0.41 |
| formal_si_sel_task | {'k': 64, 'lam0': 0.5, 'beta': 0.0} | 81.0 | 67.3 | 82.2 | 0.253 | 0.639 |  | 0.28 |
| formal_si_sel_contrast | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 82.0 | 72.9 | 84.1 | 0.273 | 0.650 |  | 0.43 |
| derom+leace | - | 77.6 | 70.3 | 77.4 | 0.262 | 0.727 |  | 0.34 |
| derom+formal_si | {'k': 16, 'lam0': 0.0, 'beta': 0.0} | 82.1 | 73.8 | 81.1 | 0.277 | 0.755 |  | 0.33 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_inv_only vs best baseline (derom_encode) | +4.2 | [-8.3, +18.0] | 0.492 | -0.022 |
| selective (best lam0>0) vs invariance-only | -1.5 | [-10.4, +6.7] | 0.754 | +0.036 |
| invariance-only vs derom_encode | +4.2 | [-8.3, +18.0] | 0.492 | -0.022 |
| derom+formal_si vs derom_encode | +3.7 | [-5.6, +14.0] | 0.422 | +0.018 |
| invariance-only vs random removal (k=16) | +7.1 | [-1.3, +17.6] | 0.092 | |

### pilot_narabizi / labse

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 77.2 | 64.7 | 74.6 | 0.203 | 0.618 |  | 0.22 |
| derom_encode | - | 77.8 | 66.9 | 72.6 | 0.276 | 0.751 |  | 0.26 |
| centering | - | 77.4 | 64.2 | 74.9 | 0.207 | 0.634 |  | 0.34 |
| lsar | {'r': 1} | 77.5 | 65.5 | 75.1 | 0.208 | 0.621 |  | 0.32 |
| leace | - | 76.3 | 63.8 | 70.1 | 0.218 | 0.639 |  | 1.13 |
| splince | - | 76.3 | 63.9 | 74.5 | 0.203 | 0.619 |  | 1.28 |
| random | {'k': 64} | 76.8 | 65.9 | 75.6 | 0.204 | 0.618 |  | 0.20 |
| formal_si_inv_only | {'k': 16, 'lam0': 0.0, 'beta': 1.0} | 75.1 | 69.0 | 71.6 | 0.250 | 0.691 |  | 1.00 |
| formal_si_sel_task | {'k': 4, 'lam0': 0.5, 'beta': 0.0} | 75.0 | 64.0 | 72.8 | 0.245 | 0.680 |  | 0.64 |
| formal_si_sel_contrast | {'k': 16, 'lam0': 0.5, 'beta': 1.0} | 77.4 | 64.1 | 73.5 | 0.274 | 0.711 |  | 1.01 |
| derom+leace | - | 75.0 | 61.2 | 70.8 | 0.278 | 0.752 |  | 0.93 |
| derom+formal_si | {'k': 16, 'lam0': 0.0, 'beta': 0.0} | 73.1 | 70.6 | 69.3 | 0.269 | 0.785 |  | 0.57 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_inv_only vs best baseline (derom_encode) | +2.0 | [-9.9, +13.6] | 0.774 | -0.026 |
| selective (best lam0>0) vs invariance-only | -4.8 | [-10.9, +1.1] | 0.124 | +0.024 |
| invariance-only vs derom_encode | +2.0 | [-9.9, +13.6] | 0.774 | -0.026 |
| derom+formal_si vs derom_encode | +3.7 | [-6.1, +13.8] | 0.442 | -0.007 |
| invariance-only vs random removal (k=16) | +7.8 | [+0.0, +15.0] | 0.050 | |

### pilot_narabizi / xlmr-base

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 65.1 | 47.4 | 63.8 | 0.159 | 0.564 |  | 0.24 |
| derom_encode | - | 66.1 | 56.0 | 68.0 | 0.285 | 0.813 |  | 0.16 |
| centering | - | 65.9 | 56.6 | 62.7 | 0.171 | 0.637 |  | 0.33 |
| lsar | {'r': 1} | 61.7 | 53.4 | 60.8 | 0.174 | 0.584 |  | 0.24 |
| leace | - | 65.2 | 46.0 | 57.5 | 0.185 | 0.650 |  | 0.83 |
| splince | - | 65.9 | 46.8 | 63.9 | 0.160 | 0.578 |  | 1.28 |
| random | {'k': 64} | 65.6 | 44.2 | 64.8 | 0.159 | 0.565 |  | 0.22 |
| formal_si_inv_only | {'k': 4, 'lam0': 0.0, 'beta': 0.0} | 67.1 | 51.2 | 67.3 | 0.193 | 0.541 |  | 0.75 |
| formal_si_sel_task | {'k': 4, 'lam0': 0.5, 'beta': 0.0} | 66.2 | 45.5 | 61.9 | 0.195 | 0.551 |  | 0.68 |
| formal_si_sel_contrast | {'k': 4, 'lam0': 0.5, 'beta': 1.0} | 66.3 | 48.3 | 63.6 | 0.221 | 0.712 |  | 0.72 |
| derom+leace | - | 61.2 | 50.0 | 54.2 | 0.292 | 0.785 |  | 0.83 |
| derom+formal_si | {'k': 4, 'lam0': 0.0, 'beta': 0.0} | 63.5 | 47.2 | 62.4 | 0.288 | 0.725 |  | 0.86 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_inv_only vs best baseline (centering) | -5.4 | [-15.5, +3.8] | 0.284 | +0.022 |
| selective (best lam0>0) vs invariance-only | -2.9 | [-9.1, +3.4] | 0.382 | +0.028 |
| invariance-only vs derom_encode | -4.8 | [-19.7, +8.7] | 0.504 | -0.092 |
| derom+formal_si vs derom_encode | -8.8 | [-21.2, +4.1] | 0.180 | +0.003 |
| invariance-only vs random removal (k=4) | +2.9 | [-3.7, +9.4] | 0.400 | |

### pilot_narabizi_topic / e5-small

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 53.8 | 29.4 | 51.0 | 0.171 | 0.479 |  | 0.44 |
| derom_encode | - | 53.0 | 31.4 | 49.3 | 0.259 | 0.722 |  | 0.38 |
| centering | - | 51.2 | 24.4 | 47.9 | 0.179 | 0.524 |  | 0.51 |
| lsar | {'r': 1} | 51.0 | 28.0 | 48.6 | 0.186 | 0.543 |  | 0.46 |
| leace | - | 52.0 | 29.6 | 46.4 | 0.184 | 0.535 |  | 0.74 |
| splince | - | 52.2 | 30.3 | 47.1 | 0.171 | 0.479 |  | 0.60 |
| random | {'k': 4} | 49.5 | 29.6 | 46.4 | 0.171 | 0.479 |  | 0.44 |
| formal_si_inv_only | {'k': 64, 'lam0': 0.0, 'beta': 1.0} | 34.2 | 30.2 | 35.9 | 0.254 | 0.641 |  | 0.60 |
| formal_si_sel_task | {'k': 64, 'lam0': 0.5, 'beta': 0.0} | 50.4 | 29.5 | 47.1 | 0.250 | 0.654 |  | 0.51 |
| formal_si_sel_contrast | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 45.6 | 32.9 | 46.0 | 0.274 | 0.662 |  | 0.59 |
| derom+leace | - | 49.9 | 29.8 | 42.8 | 0.262 | 0.727 |  | 0.60 |
| derom+formal_si | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 46.0 | 32.7 | 40.3 | 0.312 | 0.780 |  | 0.55 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_sel_contrast vs best baseline (derom_encode) | +1.5 | [-4.0, +6.9] | 0.548 | +0.015 |
| selective (best lam0>0) vs invariance-only | +2.7 | [-2.0, +7.4] | 0.256 | +0.020 |
| invariance-only vs derom_encode | -1.2 | [-7.4, +5.0] | 0.706 | -0.005 |
| derom+formal_si vs derom_encode | +1.3 | [-3.1, +5.2] | 0.568 | +0.054 |
| invariance-only vs random removal (k=64) | +1.7 | [-2.6, +6.2] | 0.408 | |

### pilot_narabizi_topic / labse

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 49.5 | 23.1 | 53.7 | 0.203 | 0.618 |  | 0.73 |
| derom_encode | - | 50.3 | 31.0 | 47.1 | 0.276 | 0.751 |  | 0.58 |
| centering | - | 51.3 | 25.7 | 48.1 | 0.207 | 0.634 |  | 0.81 |
| lsar | {'r': 1} | 50.2 | 28.5 | 54.4 | 0.208 | 0.621 |  | 0.70 |
| leace | - | 50.1 | 29.8 | 55.3 | 0.218 | 0.639 |  | 1.19 |
| splince | - | 52.2 | 25.0 | 55.8 | 0.203 | 0.618 |  | 1.78 |
| random | {'k': 4} | 48.8 | 24.3 | 52.3 | 0.203 | 0.618 |  | 0.65 |
| formal_si_inv_only | {'k': 16, 'lam0': 0.0, 'beta': 1.0} | 46.2 | 30.1 | 52.6 | 0.250 | 0.691 |  | 1.07 |
| formal_si_sel_task | {'k': 4, 'lam0': 0.5, 'beta': 0.0} | 50.2 | 26.8 | 53.8 | 0.250 | 0.684 |  | 1.07 |
| formal_si_sel_contrast | {'k': 16, 'lam0': 2.0, 'beta': 1.0} | 49.3 | 23.7 | 53.5 | 0.269 | 0.708 |  | 1.45 |
| derom+leace | - | 49.4 | 29.9 | 49.3 | 0.278 | 0.752 |  | 1.35 |
| derom+formal_si | {'k': 16, 'lam0': 0.0, 'beta': 1.0} | 41.3 | 32.1 | 39.4 | 0.269 | 0.785 |  | 1.14 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_inv_only vs best baseline (derom_encode) | -0.9 | [-9.1, +7.3] | 0.932 | -0.026 |
| selective (best lam0>0) vs invariance-only | -3.3 | [-7.3, +0.4] | 0.076 | -0.000 |
| invariance-only vs derom_encode | -0.9 | [-9.1, +7.3] | 0.932 | -0.026 |
| derom+formal_si vs derom_encode | +1.1 | [-3.4, +5.9] | 0.576 | -0.007 |
| invariance-only vs random removal (k=16) | +5.9 | [+1.8, +10.4] | 0.008 | |

### pilot_narabizi_topic / xlmr-base

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 21.5 | 12.4 | 21.3 | 0.159 | 0.564 |  | 0.37 |
| derom_encode | - | 21.7 | 18.2 | 20.0 | 0.285 | 0.813 |  | 0.40 |
| centering | - | 21.6 | 13.0 | 15.9 | 0.171 | 0.637 |  | 0.44 |
| lsar | {'r': 1} | 21.4 | 15.7 | 21.9 | 0.174 | 0.584 |  | 0.39 |
| leace | - | 22.4 | 14.8 | 20.3 | 0.185 | 0.650 |  | 1.08 |
| splince | - | 21.5 | 12.5 | 21.5 | 0.159 | 0.564 |  | 1.47 |
| random | {'k': 4} | 21.3 | 12.5 | 21.3 | 0.159 | 0.564 |  | 0.42 |
| formal_si_inv_only | {'k': 4, 'lam0': 0.0, 'beta': 0.0} | 22.6 | 19.3 | 21.7 | 0.193 | 0.541 |  | 0.88 |
| formal_si_sel_task | {'k': 4, 'lam0': 0.5, 'beta': 0.0} | 21.3 | 19.5 | 20.5 | 0.211 | 0.729 |  | 0.85 |
| formal_si_sel_contrast | {'k': 4, 'lam0': 0.5, 'beta': 1.0} | 21.0 | 20.7 | 21.2 | 0.220 | 0.719 |  | 0.96 |
| derom+leace | - | 21.8 | 15.5 | 17.7 | 0.292 | 0.785 |  | 1.04 |
| derom+formal_si | {'k': 4, 'lam0': 2.0, 'beta': 1.0} | 21.8 | 19.1 | 20.1 | 0.309 | 0.838 |  | 0.83 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_sel_contrast vs best baseline (derom_encode) | +2.5 | [-1.0, +5.7] | 0.144 | -0.065 |
| selective (best lam0>0) vs invariance-only | +1.4 | [-1.6, +4.5] | 0.404 | +0.028 |
| invariance-only vs derom_encode | +1.1 | [-2.5, +4.6] | 0.568 | -0.092 |
| derom+formal_si vs derom_encode | +0.9 | [-0.0, +2.3] | 0.110 | +0.024 |
| invariance-only vs random removal (k=4) | +6.7 | [+2.0, +11.1] | 0.004 | |

### pilot_tsac / e5-small

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 90.7 | 72.3 |  | 0.171 | 0.479 |  | 0.35 |
| derom_encode | - | 90.1 | 80.2 |  | 0.259 | 0.722 |  | 0.39 |
| centering | - | 90.4 | 72.4 |  | 0.175 | 0.522 |  | 0.55 |
| lsar | {'r': 1} | 90.1 | 72.9 |  | 0.174 | 0.490 |  | 0.39 |
| leace | - | 89.9 | 70.7 |  | 0.182 | 0.534 |  | 0.64 |
| splince | - | 90.5 | 73.5 |  | 0.171 | 0.479 |  | 0.55 |
| random | {'k': 16} | 90.0 | 72.5 |  | 0.171 | 0.478 |  | 0.35 |
| formal_si_inv_only | {'k': 64, 'lam0': 0.0, 'beta': 1.0} | 83.7 | 69.9 |  | 0.254 | 0.641 |  | 0.67 |
| formal_si_sel_task | {'k': 64, 'lam0': 0.5, 'beta': 0.0} | 88.7 | 68.5 |  | 0.253 | 0.640 |  | 0.59 |
| formal_si_sel_contrast | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 89.6 | 75.0 |  | 0.273 | 0.650 |  | 0.59 |
| derom+leace | - | 89.9 | 80.6 |  | 0.260 | 0.726 |  | 0.86 |
| derom+formal_si | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 89.6 | 81.3 |  | 0.312 | 0.770 |  | 0.59 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_sel_contrast vs best baseline (derom_encode) | -5.2 | [-9.1, -1.2] | 0.004 | +0.014 |
| selective (best lam0>0) vs invariance-only | +5.1 | [+1.4, +8.9] | 0.006 | +0.019 |
| invariance-only vs derom_encode | -10.3 | [-14.3, -6.3] | 0.000 | -0.005 |
| derom+formal_si vs derom_encode | +1.1 | [-0.5, +2.8] | 0.164 | +0.054 |
| invariance-only vs random removal (k=64) | -1.9 | [-5.5, +1.7] | 0.286 | |

### pilot_tsac / labse

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 91.8 | 71.7 |  | 0.203 | 0.618 |  | 0.46 |
| derom_encode | - | 91.1 | 71.4 |  | 0.276 | 0.751 |  | 0.45 |
| centering | - | 91.9 | 71.9 |  | 0.206 | 0.634 |  | 0.83 |
| lsar | {'r': 1} | 91.8 | 71.5 |  | 0.203 | 0.616 |  | 0.65 |
| leace | - | 91.1 | 70.5 |  | 0.214 | 0.637 |  | 1.61 |
| splince | - | 91.6 | 72.2 |  | 0.203 | 0.618 |  | 1.77 |
| random | {'k': 64} | 91.4 | 71.9 |  | 0.204 | 0.618 |  | 0.53 |
| formal_si_inv_only | {'k': 16, 'lam0': 0.0, 'beta': 1.0} | 89.4 | 72.6 |  | 0.250 | 0.691 |  | 1.00 |
| formal_si_sel_task | {'k': 4, 'lam0': 0.5, 'beta': 0.0} | 91.1 | 69.6 |  | 0.245 | 0.682 |  | 1.08 |
| formal_si_sel_contrast | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 90.7 | 69.3 |  | 0.320 | 0.737 |  | 1.21 |
| derom+leace | - | 90.6 | 73.4 |  | 0.273 | 0.749 |  | 1.86 |
| derom+formal_si | {'k': 16, 'lam0': 0.0, 'beta': 1.0} | 88.3 | 73.0 |  | 0.269 | 0.785 |  | 1.07 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_inv_only vs best baseline (splince) | +0.4 | [-2.5, +3.2] | 0.830 | +0.047 |
| selective (best lam0>0) vs invariance-only | -3.1 | [-6.0, -0.2] | 0.036 | -0.005 |
| invariance-only vs derom_encode | +1.2 | [-2.8, +5.4] | 0.542 | -0.026 |
| derom+formal_si vs derom_encode | +1.6 | [-0.8, +4.2] | 0.206 | -0.007 |
| invariance-only vs random removal (k=16) | +0.4 | [-2.4, +3.1] | 0.838 | |

### pilot_tsac / xlmr-base

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 72.2 | 46.4 |  | 0.159 | 0.564 |  | 0.97 |
| derom_encode | - | 72.1 | 47.5 |  | 0.285 | 0.813 |  | 0.50 |
| centering | - | 71.2 | 55.3 |  | 0.204 | 0.644 |  | 1.01 |
| lsar | {'r': 1} | 73.4 | 47.3 |  | 0.133 | 0.314 |  | 1.00 |
| leace | - | 72.3 | 55.4 |  | 0.184 | 0.641 |  | 2.31 |
| splince | - | 71.4 | 45.8 |  | 0.159 | 0.564 |  | 1.99 |
| random | {'k': 4} | 71.6 | 45.6 |  | 0.159 | 0.564 |  | 0.55 |
| formal_si_inv_only | {'k': 4, 'lam0': 0.0, 'beta': 0.0} | 74.0 | 58.8 |  | 0.193 | 0.541 |  | 1.05 |
| formal_si_sel_task | {'k': 4, 'lam0': 0.5, 'beta': 0.0} | 65.4 | 52.3 |  | 0.208 | 0.650 |  | 1.04 |
| formal_si_sel_contrast | {'k': 4, 'lam0': 0.5, 'beta': 1.0} | 72.6 | 56.5 |  | 0.221 | 0.713 |  | 1.50 |
| derom+leace | - | 71.0 | 60.3 |  | 0.286 | 0.772 |  | 1.65 |
| derom+formal_si | {'k': 4, 'lam0': 2.0, 'beta': 1.0} | 72.8 | 57.2 |  | 0.310 | 0.838 |  | 1.24 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_inv_only vs best baseline (leace) | +3.4 | [+0.6, +6.3] | 0.026 | +0.009 |
| selective (best lam0>0) vs invariance-only | -2.3 | [-4.6, -0.3] | 0.022 | +0.028 |
| invariance-only vs derom_encode | +11.3 | [+6.6, +16.1] | 0.000 | -0.092 |
| derom+formal_si vs derom_encode | +9.7 | [+7.3, +12.5] | 0.000 | +0.025 |
| invariance-only vs random removal (k=4) | +13.3 | [+9.9, +16.6] | 0.000 | |

## Selection rule: oracle

### pilot / e5-small

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 77.2 | 56.6 | 68.8 | 0.171 | 0.479 | 68.8 | 2.04 |
| derom_encode | - | 77.6 | 65.5 | 67.2 | 0.259 | 0.722 | 67.1 | 1.83 |
| centering | - | 77.4 | 63.6 | 59.0 | 0.176 | 0.519 | 63.5 | 6.50 |
| lsar | {'r': 1} | 77.4 | 57.0 | 69.6 | 0.172 | 0.483 | 67.4 | 2.64 |
| leace | - | 77.3 | 62.9 | 67.3 | 0.182 | 0.532 | 63.0 | 3.40 |
| splince | - | 77.0 | 56.7 | 67.9 | 0.171 | 0.479 | 68.7 | 3.04 |
| random | {'k': 64} | 76.6 | 57.5 | 66.1 | 0.170 | 0.478 | 66.8 | 2.58 |
| formal_si_inv_only | {'k': 16, 'lam0': 0.0, 'beta': 1.0} | 75.5 | 65.0 | 59.8 | 0.237 | 0.627 | 66.7 | 2.48 |
| formal_si_sel_task | {'k': 4, 'lam0': 0.5, 'beta': 0.0} | 77.3 | 52.2 | 67.9 | 0.223 | 0.602 | 69.4 | 2.65 |
| formal_si_sel_contrast | {'k': 16, 'lam0': 0.5, 'beta': 1.0} | 77.0 | 63.7 | 67.0 | 0.238 | 0.629 | 67.3 | 2.09 |
| derom+leace | - | 77.5 | 65.4 | 66.7 | 0.260 | 0.725 | 60.9 | 2.93 |
| derom+formal_si | {'k': 4, 'lam0': 0.0, 'beta': 1.0} | 77.1 | 67.1 | 70.0 | 0.268 | 0.744 | 67.2 | 2.61 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_inv_only vs best baseline (derom_encode) | -0.5 | [-3.3, +2.6] | 0.768 | -0.022 |
| selective (best lam0>0) vs invariance-only | -1.3 | [-3.2, +0.8] | 0.190 | +0.001 |
| invariance-only vs derom_encode | -0.5 | [-3.3, +2.6] | 0.768 | -0.022 |
| derom+formal_si vs derom_encode | +1.6 | [-0.2, +3.3] | 0.072 | +0.009 |
| invariance-only vs random removal (k=16) | +8.5 | [+5.8, +11.2] | 0.000 | |

### pilot / labse

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 78.8 | 63.2 | 64.6 | 0.203 | 0.618 | 72.7 | 3.29 |
| derom_encode | - | 77.9 | 62.1 | 58.5 | 0.276 | 0.751 | 70.4 | 2.96 |
| centering | - | 79.1 | 63.4 | 63.5 | 0.208 | 0.632 | 66.3 | 5.96 |
| lsar | {'r': 1} | 79.0 | 63.2 | 64.6 | 0.203 | 0.615 | 71.4 | 4.54 |
| leace | - | 79.1 | 64.3 | 66.5 | 0.213 | 0.633 | 66.8 | 7.35 |
| splince | - | 79.0 | 63.7 | 63.8 | 0.203 | 0.618 | 72.7 | 7.06 |
| random | {'k': 16} | 79.0 | 63.2 | 65.2 | 0.203 | 0.618 | 72.4 | 5.01 |
| formal_si_inv_only | {'k': 4, 'lam0': 0.0, 'beta': 1.0} | 78.9 | 66.9 | 66.2 | 0.245 | 0.682 | 70.4 | 5.71 |
| formal_si_sel_task | {'k': 4, 'lam0': 0.5, 'beta': 0.0} | 78.5 | 64.1 | 61.8 | 0.243 | 0.679 | 70.4 | 6.01 |
| formal_si_sel_contrast | {'k': 16, 'lam0': 0.5, 'beta': 1.0} | 78.7 | 67.0 | 63.2 | 0.274 | 0.711 | 69.8 | 5.11 |
| derom+leace | - | 77.9 | 60.9 | 57.9 | 0.274 | 0.748 | 62.9 | 6.61 |
| derom+formal_si | {'k': 4, 'lam0': 0.0, 'beta': 1.0} | 77.4 | 64.2 | 61.1 | 0.282 | 0.777 | 69.5 | 6.21 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_sel_contrast vs best baseline (leace) | +2.7 | [+0.8, +4.3] | 0.006 | +0.061 |
| selective (best lam0>0) vs invariance-only | +0.1 | [-1.5, +1.5] | 0.956 | +0.029 |
| invariance-only vs derom_encode | +4.8 | [+1.7, +7.9] | 0.004 | -0.031 |
| derom+formal_si vs derom_encode | +2.1 | [-0.0, +4.1] | 0.056 | +0.007 |
| invariance-only vs random removal (k=4) | +3.9 | [+1.9, +6.0] | 0.000 | |

### pilot / xlmr-base

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 70.1 | 59.8 | 57.5 | 0.159 | 0.564 | 45.3 | 3.86 |
| derom_encode | - | 69.9 | 59.5 | 49.2 | 0.285 | 0.813 | 42.9 | 3.21 |
| centering | - | 70.4 | 57.4 | 57.7 | 0.201 | 0.636 | 34.9 | 4.69 |
| lsar | {'r': 1} | 71.0 | 61.1 | 56.8 | 0.133 | 0.326 | 45.5 | 4.18 |
| leace | - | 70.4 | 57.8 | 56.0 | 0.184 | 0.629 | 35.1 | 6.42 |
| splince | - | 69.9 | 59.8 | 57.3 | 0.159 | 0.564 | 45.3 | 7.25 |
| random | {'k': 64} | 70.2 | 60.4 | 56.9 | 0.159 | 0.565 | 45.2 | 4.06 |
| formal_si_inv_only | {'k': 4, 'lam0': 0.0, 'beta': 0.0} | 70.9 | 60.9 | 55.3 | 0.193 | 0.541 | 41.8 | 4.20 |
| formal_si_sel_task | {'k': 64, 'lam0': 0.5, 'beta': 0.0} | 60.7 | 54.8 | 47.4 | 0.227 | 0.791 | 40.2 | 3.58 |
| formal_si_sel_contrast | {'k': 4, 'lam0': 0.5, 'beta': 1.0} | 70.4 | 60.7 | 54.3 | 0.219 | 0.722 | 43.3 | 4.71 |
| derom+leace | - | 70.3 | 57.6 | 51.3 | 0.290 | 0.764 | 30.4 | 5.80 |
| derom+formal_si | {'k': 16, 'lam0': 0.0, 'beta': 1.0} | 69.5 | 61.6 | 46.1 | 0.301 | 0.726 | 35.8 | 4.01 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_inv_only vs best baseline (lsar) | -0.2 | [-1.4, +0.9] | 0.674 | +0.059 |
| selective (best lam0>0) vs invariance-only | -0.2 | [-1.7, +1.3] | 0.812 | +0.026 |
| invariance-only vs derom_encode | +1.4 | [-1.6, +4.2] | 0.404 | -0.092 |
| derom+formal_si vs derom_encode | +2.1 | [-0.6, +4.6] | 0.124 | +0.016 |
| invariance-only vs random removal (k=4) | +0.6 | [-1.0, +2.2] | 0.460 | |

### pilot_myc / e5-small

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 79.2 | 51.8 |  | 0.171 | 0.479 |  | 0.66 |
| derom_encode | - | 78.8 | 66.6 |  | 0.259 | 0.722 |  | 0.68 |
| centering | - | 79.0 | 61.2 |  | 0.176 | 0.523 |  | 0.86 |
| lsar | {'r': 1} | 78.8 | 51.6 |  | 0.172 | 0.483 |  | 0.76 |
| leace | - | 79.2 | 63.6 |  | 0.185 | 0.537 |  | 1.04 |
| splince | - | 79.2 | 51.9 |  | 0.171 | 0.479 |  | 1.23 |
| random | {'k': 64} | 79.1 | 53.6 |  | 0.170 | 0.478 |  | 0.86 |
| formal_si_inv_only | {'k': 64, 'lam0': 0.0, 'beta': 1.0} | 76.4 | 65.7 |  | 0.254 | 0.641 |  | 0.84 |
| formal_si_sel_task | {'k': 64, 'lam0': 0.5, 'beta': 0.0} | 76.9 | 50.1 |  | 0.248 | 0.636 |  | 1.01 |
| formal_si_sel_contrast | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 79.3 | 66.1 |  | 0.273 | 0.654 |  | 1.09 |
| derom+leace | - | 77.9 | 64.0 |  | 0.262 | 0.729 |  | 1.56 |
| derom+formal_si | {'k': 16, 'lam0': 0.0, 'beta': 0.0} | 77.8 | 68.3 |  | 0.277 | 0.755 |  | 0.94 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_sel_contrast vs best baseline (derom_encode) | -0.5 | [-5.6, +4.7] | 0.818 | +0.014 |
| selective (best lam0>0) vs invariance-only | +0.4 | [-4.0, +5.2] | 0.810 | +0.019 |
| invariance-only vs derom_encode | -1.0 | [-5.8, +4.0] | 0.658 | -0.005 |
| derom+formal_si vs derom_encode | +1.6 | [-1.6, +5.1] | 0.310 | +0.018 |
| invariance-only vs random removal (k=64) | +12.0 | [+6.0, +17.3] | 0.000 | |

### pilot_myc / labse

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 80.2 | 68.0 |  | 0.203 | 0.618 |  | 1.25 |
| derom_encode | - | 79.8 | 58.8 |  | 0.276 | 0.751 |  | 1.34 |
| centering | - | 80.0 | 63.5 |  | 0.207 | 0.634 |  | 1.94 |
| lsar | {'r': 1} | 80.2 | 68.1 |  | 0.202 | 0.615 |  | 1.41 |
| leace | - | 80.2 | 65.5 |  | 0.216 | 0.636 |  | 2.39 |
| splince | - | 80.3 | 67.6 |  | 0.203 | 0.618 |  | 3.17 |
| random | {'k': 4} | 80.2 | 68.2 |  | 0.203 | 0.618 |  | 1.34 |
| formal_si_inv_only | {'k': 64, 'lam0': 0.0, 'beta': 1.0} | 77.4 | 69.8 |  | 0.243 | 0.709 |  | 1.98 |
| formal_si_sel_task | {'k': 4, 'lam0': 0.5, 'beta': 0.0} | 80.2 | 66.2 |  | 0.245 | 0.678 |  | 1.96 |
| formal_si_sel_contrast | {'k': 16, 'lam0': 0.5, 'beta': 1.0} | 79.8 | 70.1 |  | 0.274 | 0.711 |  | 2.26 |
| derom+leace | - | 79.1 | 60.4 |  | 0.277 | 0.753 |  | 3.22 |
| derom+formal_si | {'k': 16, 'lam0': 0.0, 'beta': 0.0} | 78.5 | 67.9 |  | 0.269 | 0.785 |  | 1.73 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_sel_contrast vs best baseline (random) | +1.9 | [-2.0, +5.7] | 0.316 | +0.071 |
| selective (best lam0>0) vs invariance-only | +0.3 | [-3.8, +4.1] | 0.866 | +0.031 |
| invariance-only vs derom_encode | +11.0 | [+4.7, +17.4] | 0.000 | -0.033 |
| derom+formal_si vs derom_encode | +9.1 | [+4.0, +14.7] | 0.000 | -0.007 |
| invariance-only vs random removal (k=64) | +1.7 | [-2.3, +5.8] | 0.412 | |

### pilot_myc / xlmr-base

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 74.3 | 60.1 |  | 0.159 | 0.564 |  | 2.13 |
| derom_encode | - | 74.0 | 59.7 |  | 0.285 | 0.813 |  | 1.41 |
| centering | - | 74.3 | 62.5 |  | 0.201 | 0.642 |  | 2.74 |
| lsar | {'r': 1} | 75.0 | 56.9 |  | 0.133 | 0.325 |  | 1.93 |
| leace | - | 74.4 | 61.5 |  | 0.193 | 0.637 |  | 3.23 |
| splince | - | 74.3 | 61.8 |  | 0.159 | 0.564 |  | 3.61 |
| random | {'k': 64} | 74.0 | 61.5 |  | 0.159 | 0.565 |  | 1.77 |
| formal_si_inv_only | {'k': 4, 'lam0': 0.0, 'beta': 0.0} | 74.8 | 65.4 |  | 0.193 | 0.541 |  | 2.20 |
| formal_si_sel_task | {'k': 64, 'lam0': 0.5, 'beta': 0.0} | 64.6 | 60.0 |  | 0.220 | 0.776 |  | 2.07 |
| formal_si_sel_contrast | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 71.3 | 65.9 |  | 0.305 | 0.814 |  | 2.51 |
| derom+leace | - | 73.5 | 64.6 |  | 0.293 | 0.789 |  | 3.01 |
| derom+formal_si | {'k': 16, 'lam0': 0.0, 'beta': 1.0} | 71.8 | 68.4 |  | 0.301 | 0.726 |  | 1.97 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_sel_contrast vs best baseline (centering) | +3.4 | [-2.3, +8.4] | 0.232 | +0.104 |
| selective (best lam0>0) vs invariance-only | +0.5 | [-4.1, +5.3] | 0.834 | +0.112 |
| invariance-only vs derom_encode | +5.7 | [+0.1, +11.4] | 0.048 | -0.092 |
| derom+formal_si vs derom_encode | +8.8 | [+3.2, +14.1] | 0.000 | +0.016 |
| invariance-only vs random removal (k=4) | +4.9 | [+1.5, +8.3] | 0.000 | |

### pilot_narabizi / e5-small

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 83.1 | 67.6 | 84.1 | 0.171 | 0.479 |  | 0.16 |
| derom_encode | - | 81.1 | 70.2 | 77.1 | 0.259 | 0.722 |  | 0.15 |
| centering | - | 83.2 | 63.4 | 83.4 | 0.179 | 0.524 |  | 0.19 |
| lsar | {'r': 1} | 82.4 | 64.3 | 84.4 | 0.186 | 0.543 |  | 0.21 |
| leace | - | 82.4 | 67.9 | 83.0 | 0.184 | 0.535 |  | 0.32 |
| splince | - | 82.9 | 68.9 | 83.8 | 0.171 | 0.479 |  | 0.32 |
| random | {'k': 4} | 84.5 | 68.1 | 84.0 | 0.171 | 0.479 |  | 0.18 |
| formal_si_inv_only | {'k': 64, 'lam0': 0.0, 'beta': 1.0} | 80.9 | 80.0 | 81.0 | 0.254 | 0.641 |  | 0.38 |
| formal_si_sel_task | {'k': 64, 'lam0': 0.5, 'beta': 0.0} | 81.0 | 67.3 | 82.2 | 0.253 | 0.639 |  | 0.28 |
| formal_si_sel_contrast | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 82.0 | 72.9 | 84.1 | 0.273 | 0.650 |  | 0.43 |
| derom+leace | - | 77.6 | 70.3 | 77.4 | 0.262 | 0.727 |  | 0.34 |
| derom+formal_si | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 81.2 | 76.8 | 82.6 | 0.312 | 0.770 |  | 0.33 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_inv_only vs best baseline (derom_encode) | +9.9 | [-2.3, +23.0] | 0.098 | -0.005 |
| selective (best lam0>0) vs invariance-only | -7.1 | [-17.6, +2.9] | 0.172 | +0.019 |
| invariance-only vs derom_encode | +9.9 | [-2.3, +23.0] | 0.098 | -0.005 |
| derom+formal_si vs derom_encode | +6.6 | [+0.8, +13.3] | 0.030 | +0.053 |
| invariance-only vs random removal (k=64) | +13.9 | [+3.9, +24.8] | 0.004 | |

### pilot_narabizi / labse

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 77.2 | 64.7 | 74.6 | 0.203 | 0.618 |  | 0.22 |
| derom_encode | - | 77.8 | 66.9 | 72.6 | 0.276 | 0.751 |  | 0.26 |
| centering | - | 77.4 | 64.2 | 74.9 | 0.207 | 0.634 |  | 0.34 |
| lsar | {'r': 1} | 77.5 | 65.5 | 75.1 | 0.208 | 0.621 |  | 0.32 |
| leace | - | 76.3 | 63.8 | 70.1 | 0.218 | 0.639 |  | 1.13 |
| splince | - | 76.3 | 63.9 | 74.5 | 0.203 | 0.619 |  | 1.28 |
| random | {'k': 64} | 76.8 | 65.9 | 75.6 | 0.204 | 0.618 |  | 0.20 |
| formal_si_inv_only | {'k': 16, 'lam0': 0.0, 'beta': 0.0} | 75.6 | 69.0 | 72.4 | 0.250 | 0.691 |  | 0.76 |
| formal_si_sel_task | {'k': 16, 'lam0': 0.5, 'beta': 0.0} | 74.8 | 64.9 | 73.4 | 0.250 | 0.691 |  | 0.62 |
| formal_si_sel_contrast | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 77.0 | 66.8 | 76.0 | 0.320 | 0.737 |  | 0.74 |
| derom+leace | - | 75.0 | 61.2 | 70.8 | 0.278 | 0.752 |  | 0.93 |
| derom+formal_si | {'k': 16, 'lam0': 0.0, 'beta': 1.0} | 74.2 | 74.3 | 70.1 | 0.269 | 0.785 |  | 0.73 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_inv_only vs best baseline (derom_encode) | +2.0 | [-9.9, +14.0] | 0.788 | -0.026 |
| selective (best lam0>0) vs invariance-only | -2.1 | [-9.2, +5.6] | 0.582 | +0.070 |
| invariance-only vs derom_encode | +2.0 | [-9.9, +14.0] | 0.788 | -0.026 |
| derom+formal_si vs derom_encode | +7.3 | [-1.9, +17.1] | 0.132 | -0.007 |
| invariance-only vs random removal (k=16) | +7.8 | [-0.4, +16.0] | 0.070 | |

### pilot_narabizi / xlmr-base

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 65.1 | 47.4 | 63.8 | 0.159 | 0.564 |  | 0.24 |
| derom_encode | - | 66.1 | 56.0 | 68.0 | 0.285 | 0.813 |  | 0.16 |
| centering | - | 65.9 | 56.6 | 62.7 | 0.171 | 0.637 |  | 0.33 |
| lsar | {'r': 1} | 61.7 | 53.4 | 60.8 | 0.174 | 0.584 |  | 0.24 |
| leace | - | 65.2 | 46.0 | 57.5 | 0.185 | 0.650 |  | 0.83 |
| splince | - | 65.9 | 46.8 | 63.9 | 0.160 | 0.578 |  | 1.28 |
| random | {'k': 4} | 66.8 | 48.3 | 61.6 | 0.159 | 0.564 |  | 0.21 |
| formal_si_inv_only | {'k': 4, 'lam0': 0.0, 'beta': 1.0} | 64.7 | 51.4 | 66.8 | 0.193 | 0.541 |  | 0.87 |
| formal_si_sel_task | {'k': 64, 'lam0': 2.0, 'beta': 0.0} | 62.9 | 47.3 | 63.9 | 0.231 | 0.606 |  | 0.63 |
| formal_si_sel_contrast | {'k': 4, 'lam0': 0.5, 'beta': 1.0} | 66.3 | 48.3 | 63.6 | 0.221 | 0.712 |  | 0.72 |
| derom+leace | - | 61.2 | 50.0 | 54.2 | 0.292 | 0.785 |  | 0.83 |
| derom+formal_si | {'k': 4, 'lam0': 2.0, 'beta': 1.0} | 68.3 | 62.5 | 68.0 | 0.310 | 0.838 |  | 0.61 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_inv_only vs best baseline (centering) | -5.2 | [-16.1, +5.2] | 0.340 | +0.022 |
| selective (best lam0>0) vs invariance-only | -3.1 | [-9.2, +3.3] | 0.354 | +0.028 |
| invariance-only vs derom_encode | -4.6 | [-19.4, +9.5] | 0.546 | -0.092 |
| derom+formal_si vs derom_encode | +6.5 | [-1.3, +15.7] | 0.112 | +0.025 |
| invariance-only vs random removal (k=4) | +3.1 | [-3.7, +9.5] | 0.398 | |

### pilot_narabizi_topic / e5-small

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 53.8 | 29.4 | 51.0 | 0.171 | 0.479 |  | 0.44 |
| derom_encode | - | 53.0 | 31.4 | 49.3 | 0.259 | 0.722 |  | 0.38 |
| centering | - | 51.2 | 24.4 | 47.9 | 0.179 | 0.524 |  | 0.51 |
| lsar | {'r': 1} | 51.0 | 28.0 | 48.6 | 0.186 | 0.543 |  | 0.46 |
| leace | - | 52.0 | 29.6 | 46.4 | 0.184 | 0.535 |  | 0.74 |
| splince | - | 52.2 | 30.3 | 47.1 | 0.171 | 0.479 |  | 0.60 |
| random | {'k': 16} | 52.7 | 29.7 | 49.5 | 0.171 | 0.478 |  | 0.46 |
| formal_si_inv_only | {'k': 16, 'lam0': 0.0, 'beta': 1.0} | 46.3 | 33.8 | 41.6 | 0.237 | 0.627 |  | 0.53 |
| formal_si_sel_task | {'k': 4, 'lam0': 0.5, 'beta': 0.0} | 50.3 | 31.0 | 48.1 | 0.223 | 0.604 |  | 0.55 |
| formal_si_sel_contrast | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 45.6 | 32.9 | 46.0 | 0.274 | 0.662 |  | 0.59 |
| derom+leace | - | 49.9 | 29.8 | 42.8 | 0.262 | 0.727 |  | 0.60 |
| derom+formal_si | {'k': 64, 'lam0': 0.5, 'beta': 0.0} | 51.7 | 34.4 | 48.3 | 0.282 | 0.772 |  | 0.47 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_inv_only vs best baseline (derom_encode) | +2.4 | [-3.5, +8.4] | 0.442 | -0.022 |
| selective (best lam0>0) vs invariance-only | -0.9 | [-4.0, +1.8] | 0.604 | +0.037 |
| invariance-only vs derom_encode | +2.4 | [-3.5, +8.4] | 0.442 | -0.022 |
| derom+formal_si vs derom_encode | +3.0 | [-0.6, +7.2] | 0.118 | +0.023 |
| invariance-only vs random removal (k=16) | +4.1 | [+1.1, +7.5] | 0.002 | |

### pilot_narabizi_topic / labse

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 49.5 | 23.1 | 53.7 | 0.203 | 0.618 |  | 0.73 |
| derom_encode | - | 50.3 | 31.0 | 47.1 | 0.276 | 0.751 |  | 0.58 |
| centering | - | 51.3 | 25.7 | 48.1 | 0.207 | 0.634 |  | 0.81 |
| lsar | {'r': 1} | 50.2 | 28.5 | 54.4 | 0.208 | 0.621 |  | 0.70 |
| leace | - | 50.1 | 29.8 | 55.3 | 0.218 | 0.639 |  | 1.19 |
| splince | - | 52.2 | 25.0 | 55.8 | 0.203 | 0.618 |  | 1.78 |
| random | {'k': 64} | 50.6 | 24.7 | 54.4 | 0.204 | 0.618 |  | 0.63 |
| formal_si_inv_only | {'k': 4, 'lam0': 0.0, 'beta': 0.0} | 46.5 | 31.0 | 53.3 | 0.245 | 0.682 |  | 1.42 |
| formal_si_sel_task | {'k': 64, 'lam0': 0.5, 'beta': 0.0} | 51.1 | 29.9 | 58.5 | 0.253 | 0.720 |  | 1.26 |
| formal_si_sel_contrast | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 49.0 | 28.6 | 55.6 | 0.322 | 0.743 |  | 1.24 |
| derom+leace | - | 49.4 | 29.9 | 49.3 | 0.278 | 0.752 |  | 1.35 |
| derom+formal_si | {'k': 64, 'lam0': 2.0, 'beta': 1.0} | 49.6 | 35.1 | 46.9 | 0.323 | 0.800 |  | 1.54 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_inv_only vs best baseline (derom_encode) | -0.0 | [-9.1, +9.1] | 0.970 | -0.031 |
| selective (best lam0>0) vs invariance-only | -1.1 | [-6.4, +4.0] | 0.776 | +0.009 |
| invariance-only vs derom_encode | -0.0 | [-9.1, +9.1] | 0.970 | -0.031 |
| derom+formal_si vs derom_encode | +4.0 | [-0.4, +8.9] | 0.076 | +0.048 |
| invariance-only vs random removal (k=4) | +6.7 | [+1.0, +12.6] | 0.026 | |

### pilot_narabizi_topic / xlmr-base

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 21.5 | 12.4 | 21.3 | 0.159 | 0.564 |  | 0.37 |
| derom_encode | - | 21.7 | 18.2 | 20.0 | 0.285 | 0.813 |  | 0.40 |
| centering | - | 21.6 | 13.0 | 15.9 | 0.171 | 0.637 |  | 0.44 |
| lsar | {'r': 1} | 21.4 | 15.7 | 21.9 | 0.174 | 0.584 |  | 0.39 |
| leace | - | 22.4 | 14.8 | 20.3 | 0.185 | 0.650 |  | 1.08 |
| splince | - | 21.5 | 12.5 | 21.5 | 0.159 | 0.564 |  | 1.47 |
| random | {'k': 64} | 21.4 | 13.1 | 21.3 | 0.159 | 0.565 |  | 0.42 |
| formal_si_inv_only | {'k': 16, 'lam0': 0.0, 'beta': 0.0} | 23.1 | 19.6 | 21.7 | 0.200 | 0.538 |  | 0.86 |
| formal_si_sel_task | {'k': 4, 'lam0': 0.5, 'beta': 0.0} | 21.3 | 19.5 | 20.5 | 0.211 | 0.729 |  | 0.85 |
| formal_si_sel_contrast | {'k': 4, 'lam0': 0.5, 'beta': 1.0} | 21.0 | 20.7 | 21.2 | 0.220 | 0.719 |  | 0.96 |
| derom+leace | - | 21.8 | 15.5 | 17.7 | 0.292 | 0.785 |  | 1.04 |
| derom+formal_si | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 20.5 | 20.5 | 20.6 | 0.381 | 0.879 |  | 1.02 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_sel_contrast vs best baseline (derom_encode) | +2.5 | [-1.0, +5.7] | 0.144 | -0.065 |
| selective (best lam0>0) vs invariance-only | +1.0 | [-1.6, +3.7] | 0.442 | +0.020 |
| invariance-only vs derom_encode | +1.5 | [-1.4, +4.1] | 0.284 | -0.085 |
| derom+formal_si vs derom_encode | +2.4 | [+0.4, +4.5] | 0.012 | +0.096 |
| invariance-only vs random removal (k=16) | +7.0 | [+2.4, +11.6] | 0.006 | |

### pilot_tsac / e5-small

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 90.7 | 72.3 |  | 0.171 | 0.479 |  | 0.35 |
| derom_encode | - | 90.1 | 80.2 |  | 0.259 | 0.722 |  | 0.39 |
| centering | - | 90.4 | 72.4 |  | 0.175 | 0.522 |  | 0.55 |
| lsar | {'r': 1} | 90.1 | 72.9 |  | 0.174 | 0.490 |  | 0.39 |
| leace | - | 89.9 | 70.7 |  | 0.182 | 0.534 |  | 0.64 |
| splince | - | 90.5 | 73.5 |  | 0.171 | 0.479 |  | 0.55 |
| random | {'k': 4} | 90.5 | 73.0 |  | 0.171 | 0.479 |  | 0.33 |
| formal_si_inv_only | {'k': 16, 'lam0': 0.0, 'beta': 0.0} | 88.6 | 77.0 |  | 0.237 | 0.627 |  | 0.62 |
| formal_si_sel_task | {'k': 64, 'lam0': 0.5, 'beta': 0.0} | 88.7 | 68.5 |  | 0.253 | 0.640 |  | 0.59 |
| formal_si_sel_contrast | {'k': 64, 'lam0': 0.5, 'beta': 1.0} | 89.6 | 75.0 |  | 0.273 | 0.650 |  | 0.59 |
| derom+leace | - | 89.9 | 80.6 |  | 0.260 | 0.726 |  | 0.86 |
| derom+formal_si | {'k': 16, 'lam0': 0.5, 'beta': 0.0} | 89.5 | 81.5 |  | 0.277 | 0.755 |  | 0.56 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_inv_only vs best baseline (derom_encode) | -3.2 | [-6.8, +0.4] | 0.110 | -0.022 |
| selective (best lam0>0) vs invariance-only | -2.0 | [-5.1, +1.0] | 0.188 | +0.036 |
| invariance-only vs derom_encode | -3.2 | [-6.8, +0.4] | 0.110 | -0.022 |
| derom+formal_si vs derom_encode | +1.3 | [+0.0, +2.9] | 0.046 | +0.018 |
| invariance-only vs random removal (k=16) | +4.5 | [+1.4, +7.3] | 0.002 | |

### pilot_tsac / labse

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 91.8 | 71.7 |  | 0.203 | 0.618 |  | 0.46 |
| derom_encode | - | 91.1 | 71.4 |  | 0.276 | 0.751 |  | 0.45 |
| centering | - | 91.9 | 71.9 |  | 0.206 | 0.634 |  | 0.83 |
| lsar | {'r': 1} | 91.8 | 71.5 |  | 0.203 | 0.616 |  | 0.65 |
| leace | - | 91.1 | 70.5 |  | 0.214 | 0.637 |  | 1.61 |
| splince | - | 91.6 | 72.2 |  | 0.203 | 0.618 |  | 1.77 |
| random | {'k': 16} | 91.6 | 72.2 |  | 0.203 | 0.618 |  | 0.48 |
| formal_si_inv_only | {'k': 16, 'lam0': 0.0, 'beta': 1.0} | 89.4 | 72.6 |  | 0.250 | 0.691 |  | 1.00 |
| formal_si_sel_task | {'k': 4, 'lam0': 2.0, 'beta': 0.0} | 90.7 | 70.2 |  | 0.245 | 0.682 |  | 1.02 |
| formal_si_sel_contrast | {'k': 16, 'lam0': 0.5, 'beta': 1.0} | 90.8 | 70.7 |  | 0.274 | 0.711 |  | 1.19 |
| derom+leace | - | 90.6 | 73.4 |  | 0.273 | 0.749 |  | 1.86 |
| derom+formal_si | {'k': 16, 'lam0': 0.0, 'beta': 0.0} | 88.3 | 73.9 |  | 0.269 | 0.785 |  | 1.08 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_inv_only vs best baseline (random) | +0.4 | [-2.4, +3.1] | 0.838 | +0.047 |
| selective (best lam0>0) vs invariance-only | -2.0 | [-4.7, +0.6] | 0.142 | +0.024 |
| invariance-only vs derom_encode | +1.2 | [-2.8, +5.4] | 0.542 | -0.026 |
| derom+formal_si vs derom_encode | +2.5 | [-0.0, +5.0] | 0.052 | -0.007 |
| invariance-only vs random removal (k=16) | +0.4 | [-2.4, +3.1] | 0.838 | |

### pilot_tsac / xlmr-base

| family | chosen setting | in | cross | codesw | AUC | inv | dialect | fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | - | 72.2 | 46.4 |  | 0.159 | 0.564 |  | 0.97 |
| derom_encode | - | 72.1 | 47.5 |  | 0.285 | 0.813 |  | 0.50 |
| centering | - | 71.2 | 55.3 |  | 0.204 | 0.644 |  | 1.01 |
| lsar | {'r': 1} | 73.4 | 47.3 |  | 0.133 | 0.314 |  | 1.00 |
| leace | - | 72.3 | 55.4 |  | 0.184 | 0.641 |  | 2.31 |
| splince | - | 71.4 | 45.8 |  | 0.159 | 0.564 |  | 1.99 |
| random | {'k': 4} | 71.6 | 45.6 |  | 0.159 | 0.564 |  | 0.55 |
| formal_si_inv_only | {'k': 4, 'lam0': 0.0, 'beta': 1.0} | 73.7 | 59.2 |  | 0.193 | 0.541 |  | 0.93 |
| formal_si_sel_task | {'k': 4, 'lam0': 2.0, 'beta': 0.0} | 64.9 | 53.2 |  | 0.208 | 0.658 |  | 1.25 |
| formal_si_sel_contrast | {'k': 4, 'lam0': 2.0, 'beta': 1.0} | 72.4 | 58.1 |  | 0.212 | 0.729 |  | 1.38 |
| derom+leace | - | 71.0 | 60.3 |  | 0.286 | 0.772 |  | 1.65 |
| derom+formal_si | {'k': 4, 'lam0': 0.0, 'beta': 1.0} | 73.6 | 63.4 |  | 0.288 | 0.725 |  | 1.21 |

| comparison (cross-script) | delta F1 | 95% CI | p | d AUC |
| --- | ---: | --- | ---: | ---: |
| formal_si_inv_only vs best baseline (leace) | +3.7 | [+0.9, +6.7] | 0.010 | +0.009 |
| selective (best lam0>0) vs invariance-only | -1.1 | [-3.0, +0.5] | 0.226 | +0.020 |
| invariance-only vs derom_encode | +11.7 | [+7.0, +16.4] | 0.000 | -0.092 |
| derom+formal_si vs derom_encode | +15.9 | [+12.4, +19.4] | 0.000 | +0.003 |
| invariance-only vs random removal (k=4) | +13.6 | [+10.3, +16.9] | 0.000 | |

## Holm-corrected: FORMAL-SI vs best baseline (loco)

| cell | delta | p | p_holm | reject |
| --- | ---: | ---: | ---: | --- |
| pilot_tsac|e5-small|formal_si_sel_contrast vs best baseline (derom_encode) | -5.2 | 0.004 | 0.048 | True |
| pilot_tsac|xlmr-base|formal_si_inv_only vs best baseline (leace) | +3.4 | 0.026 | 0.286 | False |
| pilot_narabizi_topic|xlmr-base|formal_si_sel_contrast vs best baseline (derom_encode) | +2.5 | 0.144 | 1.000 | False |
| pilot_myc|xlmr-base|formal_si_sel_contrast vs best baseline (centering) | +3.2 | 0.258 | 1.000 | False |
| pilot_narabizi|xlmr-base|formal_si_inv_only vs best baseline (centering) | -5.4 | 0.284 | 1.000 | False |
| pilot_myc|labse|formal_si_inv_only vs best baseline (lsar) | +1.6 | 0.438 | 1.000 | False |
| pilot_narabizi|e5-small|formal_si_inv_only vs best baseline (derom_encode) | +4.2 | 0.492 | 1.000 | False |
| pilot_narabizi_topic|e5-small|formal_si_sel_contrast vs best baseline (derom_encode) | +1.5 | 0.548 | 1.000 | False |
| pilot_narabizi|labse|formal_si_inv_only vs best baseline (derom_encode) | +2.0 | 0.774 | 1.000 | False |
| pilot_myc|e5-small|formal_si_sel_contrast vs best baseline (derom_encode) | -0.5 | 0.818 | 1.000 | False |
| pilot_tsac|labse|formal_si_inv_only vs best baseline (splince) | +0.4 | 0.830 | 1.000 | False |
| pilot_narabizi_topic|labse|formal_si_inv_only vs best baseline (derom_encode) | -0.9 | 0.932 | 1.000 | False |

## Holm-corrected: FORMAL-SI vs best baseline (oracle)

| cell | delta | p | p_holm | reject |
| --- | ---: | ---: | ---: | --- |
| pilot|labse|formal_si_sel_contrast vs best baseline (leace) | +2.7 | 0.006 | 0.090 | False |
| pilot_tsac|xlmr-base|formal_si_inv_only vs best baseline (leace) | +3.7 | 0.010 | 0.140 | False |
| pilot_narabizi|e5-small|formal_si_inv_only vs best baseline (derom_encode) | +9.9 | 0.098 | 1.000 | False |
| pilot_tsac|e5-small|formal_si_inv_only vs best baseline (derom_encode) | -3.2 | 0.110 | 1.000 | False |
| pilot_narabizi_topic|xlmr-base|formal_si_sel_contrast vs best baseline (derom_encode) | +2.5 | 0.144 | 1.000 | False |
| pilot_myc|xlmr-base|formal_si_sel_contrast vs best baseline (centering) | +3.4 | 0.232 | 1.000 | False |
| pilot_myc|labse|formal_si_sel_contrast vs best baseline (random) | +1.9 | 0.316 | 1.000 | False |
| pilot_narabizi|xlmr-base|formal_si_inv_only vs best baseline (centering) | -5.2 | 0.340 | 1.000 | False |
| pilot_narabizi_topic|e5-small|formal_si_inv_only vs best baseline (derom_encode) | +2.4 | 0.442 | 1.000 | False |
| pilot|xlmr-base|formal_si_inv_only vs best baseline (lsar) | -0.2 | 0.674 | 1.000 | False |
| pilot|e5-small|formal_si_inv_only vs best baseline (derom_encode) | -0.5 | 0.768 | 1.000 | False |
| pilot_narabizi|labse|formal_si_inv_only vs best baseline (derom_encode) | +2.0 | 0.788 | 1.000 | False |
| pilot_myc|e5-small|formal_si_sel_contrast vs best baseline (derom_encode) | -0.5 | 0.818 | 1.000 | False |
| pilot_tsac|labse|formal_si_inv_only vs best baseline (random) | +0.4 | 0.838 | 1.000 | False |
| pilot_narabizi_topic|labse|formal_si_inv_only vs best baseline (derom_encode) | -0.0 | 0.970 | 1.000 | False |

