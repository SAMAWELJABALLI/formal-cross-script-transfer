G1 problem exists: PASS (gap left after the better of raw and deromanize-then-encode: [11.67, 15.66, 10.36])
G2 trade-off exists: PASS
G3 method helps (point estimate; confirm with grouped bootstrap): PASS
Verdict on dev data: GO

```json
{
  "e5-small": {
    "raw_gap_points": 20.54,
    "derom_residual_gap_points": 11.67,
    "g2_tradeoff": true,
    "best_formal_si": "('formal_si', (('beta', 1.0), ('k', 16), ('lam0', 0.0)))",
    "best_unconditional": "('derom_encode', ())",
    "gain_points": -0.45,
    "formal_si_auc_loss": -0.0658,
    "g3_pass_point_estimate": false
  },
  "labse": {
    "raw_gap_points": 15.66,
    "derom_residual_gap_points": 16.74,
    "g2_tradeoff": true,
    "best_formal_si": "('formal_si', (('beta', 1.0), ('k', 16), ('lam0', 0.5)))",
    "best_unconditional": "('leace', ())",
    "gain_points": 2.66,
    "formal_si_auc_loss": -0.0708,
    "g3_pass_point_estimate": true
  },
  "xlmr-base": {
    "raw_gap_points": 10.36,
    "derom_residual_gap_points": 10.59,
    "g2_tradeoff": false,
    "best_formal_si": "('formal_si', (('beta', 0.0), ('k', 4), ('lam0', 0.0)))",
    "best_unconditional": "('lsar', (('r', 1),))",
    "gain_points": -0.18,
    "formal_si_auc_loss": -0.0338,
    "g3_pass_point_estimate": false
  }
}
```
