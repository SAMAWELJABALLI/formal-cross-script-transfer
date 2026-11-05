G1 problem exists: PASS (gap left after the better of raw and deromanize-then-encode: [10.47, 20.15, 24.75])
G2 trade-off exists: FAIL
G3 method helps (point estimate; confirm with grouped bootstrap): PASS
Verdict on dev data: NO-GO

```json
{
  "e5-small": {
    "raw_gap_points": 18.37,
    "derom_residual_gap_points": 10.47,
    "g2_tradeoff": false,
    "best_formal_si": "('formal_si', (('beta', 0.0), ('k', 16), ('lam0', 0.0)))",
    "best_unconditional": "('derom_encode', ())",
    "gain_points": -3.21,
    "formal_si_auc_loss": -0.0658,
    "g3_pass_point_estimate": false
  },
  "labse": {
    "raw_gap_points": 20.15,
    "derom_residual_gap_points": 20.44,
    "g2_tradeoff": false,
    "best_formal_si": "('formal_si', (('beta', 1.0), ('k', 16), ('lam0', 0.0)))",
    "best_unconditional": "('splince', ())",
    "gain_points": 0.4,
    "formal_si_auc_loss": -0.047,
    "g3_pass_point_estimate": false
  },
  "xlmr-base": {
    "raw_gap_points": 25.88,
    "derom_residual_gap_points": 24.75,
    "g2_tradeoff": false,
    "best_formal_si": "('formal_si', (('beta', 1.0), ('k', 4), ('lam0', 0.0)))",
    "best_unconditional": "('leace', ())",
    "gain_points": 3.74,
    "formal_si_auc_loss": -0.0338,
    "g3_pass_point_estimate": true
  }
}
```
