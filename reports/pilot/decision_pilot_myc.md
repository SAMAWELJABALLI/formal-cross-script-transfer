G1 problem exists: PASS (gap left after the better of raw and deromanize-then-encode: [12.56, 12.15, 14.19])
G2 trade-off exists: FAIL
G3 method helps (point estimate; confirm with grouped bootstrap): PASS
Verdict on dev data: NO-GO

```json
{
  "e5-small": {
    "raw_gap_points": 27.35,
    "derom_residual_gap_points": 12.56,
    "g2_tradeoff": false,
    "best_formal_si": "('formal_si', (('beta', 1.0), ('k', 64), ('lam0', 0.5)))",
    "best_unconditional": "('derom_encode', ())",
    "gain_points": -0.52,
    "formal_si_auc_loss": -0.102,
    "g3_pass_point_estimate": false
  },
  "labse": {
    "raw_gap_points": 12.15,
    "derom_residual_gap_points": 21.41,
    "g2_tradeoff": false,
    "best_formal_si": "('formal_si', (('beta', 1.0), ('k', 16), ('lam0', 0.5)))",
    "best_unconditional": "('lsar', (('r', 1),))",
    "gain_points": 2.0,
    "formal_si_auc_loss": -0.0705,
    "g3_pass_point_estimate": false
  },
  "xlmr-base": {
    "raw_gap_points": 14.19,
    "derom_residual_gap_points": 14.69,
    "g2_tradeoff": false,
    "best_formal_si": "('formal_si', (('beta', 1.0), ('k', 64), ('lam0', 0.5)))",
    "best_unconditional": "('centering', ())",
    "gain_points": 3.39,
    "formal_si_auc_loss": -0.1459,
    "g3_pass_point_estimate": true
  }
}
```
