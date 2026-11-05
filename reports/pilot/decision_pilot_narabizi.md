G1 problem exists: PASS (gap left after the better of raw and deromanize-then-encode: [12.98, 10.3, 9.07])
G2 trade-off exists: FAIL
G3 method helps (point estimate; confirm with grouped bootstrap): PASS
Verdict on dev data: NO-GO

```json
{
  "e5-small": {
    "raw_gap_points": 15.53,
    "derom_residual_gap_points": 12.98,
    "g2_tradeoff": false,
    "best_formal_si": "('formal_si', (('beta', 1.0), ('k', 64), ('lam0', 0.0)))",
    "best_unconditional": "('derom_encode', ())",
    "gain_points": 9.86,
    "formal_si_auc_loss": -0.0832,
    "g3_pass_point_estimate": true
  },
  "labse": {
    "raw_gap_points": 12.58,
    "derom_residual_gap_points": 10.3,
    "g2_tradeoff": false,
    "best_formal_si": "('formal_si', (('beta', 0.0), ('k', 16), ('lam0', 0.0)))",
    "best_unconditional": "('derom_encode', ())",
    "gain_points": 2.04,
    "formal_si_auc_loss": -0.047,
    "g3_pass_point_estimate": true
  },
  "xlmr-base": {
    "raw_gap_points": 17.64,
    "derom_residual_gap_points": 9.07,
    "g2_tradeoff": false,
    "best_formal_si": "('formal_si', (('beta', 1.0), ('k', 4), ('lam0', 0.0)))",
    "best_unconditional": "('centering', ())",
    "gain_points": -5.17,
    "formal_si_auc_loss": -0.0338,
    "g3_pass_point_estimate": false
  }
}
```
