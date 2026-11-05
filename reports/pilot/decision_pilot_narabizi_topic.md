G1 problem exists: PASS (gap left after the better of raw and deromanize-then-encode: [22.36, 18.49, 3.35])
G2 trade-off exists: FAIL
G3 method helps (point estimate; confirm with grouped bootstrap): PASS
Verdict on dev data: NO-GO

```json
{
  "e5-small": {
    "raw_gap_points": 24.36,
    "derom_residual_gap_points": 22.36,
    "g2_tradeoff": false,
    "best_formal_si": "('formal_si', (('beta', 1.0), ('k', 16), ('lam0', 0.0)))",
    "best_unconditional": "('derom_encode', ())",
    "gain_points": 2.37,
    "formal_si_auc_loss": -0.0658,
    "g3_pass_point_estimate": true
  },
  "labse": {
    "raw_gap_points": 26.37,
    "derom_residual_gap_points": 18.49,
    "g2_tradeoff": false,
    "best_formal_si": "('formal_si', (('beta', 0.0), ('k', 4), ('lam0', 0.0)))",
    "best_unconditional": "('derom_encode', ())",
    "gain_points": -0.03,
    "formal_si_auc_loss": -0.0415,
    "g3_pass_point_estimate": false
  },
  "xlmr-base": {
    "raw_gap_points": 9.1,
    "derom_residual_gap_points": 3.35,
    "g2_tradeoff": false,
    "best_formal_si": "('formal_si', (('beta', 1.0), ('k', 4), ('lam0', 0.5)))",
    "best_unconditional": "('derom_encode', ())",
    "gain_points": 2.5,
    "formal_si_auc_loss": -0.0613,
    "g3_pass_point_estimate": true
  }
}
```
