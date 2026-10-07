
# Light GBM Baseline

**train rows:** 1,000,000  
**val rows:** 454,045  | **val blunder rate:** 0.0969
**constant-prediction log loss:** 0.3182
| Model | No. Features | PR-AUC | ROC-AUC | Log_loss |
| ----- | ------- | ------ | ------- | -------- |
| A: rating + ply | 5 | 0.1382 | 0.6134 | 0.3104 |
| B: A + clock | 8 | 0.1409 | 0.6184 | 0.3097 |
| C: A + board | 28 | 0.2076 | 0.7225 | 0.2892 |
| D: A + board + clock | 31 | 0.2073 | 0.7224 | 0.2892 |

## Log-loss gain from adding clock features (+ve = clock helps; 95% CI resamples games)

| Comparison | Clock | No. Rows | Log Loss Gain | Confidence Interval |
| ---------- | ----- | -------- | ------------- | ------------------- |
| A->B (no board) | all rows | 454,045 | +0.00068 | [+0.00052, +0.00085] |
| A->B (no board) | clock <= 120s | 22,894 | +0.00127 | [-0.00002, +0.00257] |
| A->B (no board) | clock <= 60s | 12,156 | +0.00256 | [+0.00041, +0.00479] |
| C->D (with board) | all rows | 454,045 | -0.00000 | [-0.00015, +0.00015] |
| C->D (with board) | clock <= 120s | 22,894 | +0.00189 | [+0.00044, +0.00331] |
| C->D (with board) | clock <= 60s | 12,156 | +0.00393 | [+0.00151, +0.00637] |

## Model D by win% before the move (is it just detecting 'steep part of the curve'?)

| Win Prob Before | No. Rows | Base Rate | PR-AUC | Lift |
| --------------- | -------- | --------- | ------ | ---- |
| (0.149, 0.35] | 54,856 | 0.0804 | 0.1520 | 1.89x |
| (0.35, 0.65] | 220,345 | 0.0954 | 0.2181 | 2.29x |
| (0.65, 0.85] | 65,377 | 0.1756 | 0.2708 | 1.54x |
| (0.85, 1.0] | 113,467 | 0.0623 | 0.1765 | 2.83x |

## Model D Top features:

| Feature | Gain |
| ------- | ---- |
| material_diff | 0.189 |
| mover_elo | 0.131 |
| opp_elo | 0.069 |
| ply | 0.053 |
| my_attacked | 0.052 |
| clock_before | 0.049 |
| my_king_zone_attacked | 0.048 |
| opp_clock | 0.047 |
| n_legal | 0.047 |
| my_p | 0.034 |
| total_material | 0.030 |
| opp_king_zone_attacked | 0.028 |

# XGBoost Comparison

| Feature set | Model | PR-AUC | ROC-AUC | Log Loss | Seconds |
| ----------- | ----- | ------ | ------- | -------- | ------- |
| A: rating + ply | LightGBM | 0.1382 | 0.6134 | 0.3104 | 3 |
| A: rating + ply | XGBoost | 0.1385 | 0.6139 | 0.3103 | 3 |
| B: A + clock | LightGBM | 0.1409 | 0.6184 | 0.3097 | 3 |
| B: A + clock | XGBoost | 0.1410 | 0.6183 | 0.3097 | 3 |
| C: A + board | LightGBM | 0.2076 | 0.7225 | 0.2892 | 12 |
| C: A + board | XGBoost | 0.2068 | 0.7224 | 0.2893 | 24 |
| D: A + board + clock | LightGBM | 0.2073 | 0.7224 | 0.2892 | 12 |
| D: A + board + clock | XGBoost | 0.2075 | 0.7224 | 0.2892 | 20 |

## Model D: log-loss gain of XGBoost over LightGBM
| Clock filter | No. Rows | Gain | Confidence Interval |
| ------------ | -------- | ---- | ------------------- |
| all rows | 454,045 | +0.00004 | [-0.00007, +0.00015] |
| clock <= 60s | 12,156 | +0.00022 | [-0.00103, +0.00146] |