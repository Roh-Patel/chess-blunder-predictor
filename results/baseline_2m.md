**train rows:** 2,000,000 ; **val rows:** 454,045 ; **val blunder rate:** 0.0969
**constant-prediction log loss:** 0.3182
| Model | Num_feats | PR-AUC | ROC-AUC | Log_loss |
| ----- | --------- | ------ | ------- | -------- |
| A: rating + ply | 5 | 0.1388 | 0.6140 | 0.3102 |
| B: A + clock | 8 | 0.1415 | 0.6189 | 0.3096 |
| C: A + board | 28 | 0.2088 | 0.7240 | 0.2888 |
| D: A + board + clock | 31 | 0.2088 | 0.7238 | 0.2888 |

### Log-loss gain from adding clock features (positive = clock helps; 95% CI resamples games)
| Comparison | Clock filter | No. Rows | Gain | Confidence Interval |
| ---------- | ------------ | -------- | ---- | ------------------- |
| A->B (no board) | all rows | 454,045 | +0.00065 | [+0.00050, +0.00082] |
| A->B (no board) | clock <= 120s | 22,894 | +0.00133 | [+0.00008, +0.00275] |
| A->B (no board) | clock <= 60s | 12,156 | +0.00247 | [+0.00035, +0.00463] |
| C->D (with board) | all rows | 454,045 | +0.00000 | [-0.00014, +0.00015] |
| C->D (with board) | clock <= 120s | 22,894 | +0.00165 | [+0.00030, +0.00307] |
| C->D (with board) | clock <= 60s | 12,156 | +0.00379 | [+0.00134, +0.00648] |

### Model D by win probability before the move (is it just detecting 'steep part of the probability curve'?)
| WP Before | No. Rows | Base Rate | PR-AUC | Lift |
| --------- | -------- | --------- | ------ | ---- |
| (0.149, 0.35] | 54,856 | 0.0804 | 0.1526 | 1.90x |
| (0.35, 0.65] | 220,345 | 0.0954 | 0.2196 | 2.30x |
| (0.65, 0.85] | 65,377 | 0.1756 | 0.2707 | 1.54x |
| (0.85, 1.0] | 113,467 | 0.0623 | 0.1793 | 2.88x |

### Top features (model D, gain):
| Feature | Gain |
| ------- | ---- |
| material_diff | 0.199 |
| mover_elo | 0.133 |
| opp_elo | 0.066 |
| ply | 0.060 |
| my_attacked | 0.055 |
| my_king_zone_attacked | 0.048 |
| n_legal | 0.047 |
| clock_before | 0.045 |
| opp_clock | 0.040 |
| my_p | 0.034 |
| opp_king_zone_attacked | 0.029 |
| opp_q | 0.028 |