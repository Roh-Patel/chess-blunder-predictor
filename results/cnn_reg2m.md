**device:** cuda ; **clock features:** True ; **hybrid:** False ; **Tag:** reg2m
**train rows:** 2,000,000 ; **val rows:** 454,045 ; **val blunder rate:** 0.0969
**parameters:** 1,006,785 ; **scalar inputs:** 10
| Epoch | Train Loss | Val Log Loss | Val PR-AUC | Time (s) |
| ----- | ---------- | ------------ | ---------- | -------- |
| 1 | 0.3046 | 0.2987 | 0.1804 | 36 |
| 2 | 0.2974 | 0.2958 | 0.1872 | 23 |
| 3 | 0.2951 | 0.2946 | 0.1896 | 25 |
| 4 | 0.2926 | 0.2947 | 0.1926 | 32 |
| 5 | 0.2902 | 0.2919 | 0.1999 | 32 |
| 6 | 0.2875 | 0.2891 | 0.2072 | 32 |
| 7 | 0.2843 | 0.2878 | 0.2099 | 31 |
| 8 | 0.2802 | 0.2870 | 0.2136 | 24 |
| 9 | 0.2755 | 0.2871 | 0.2159 | 23 |
| 10 | 0.2720 | 0.2879 | 0.2155 | 23 |

**constant-prediction log loss: 0.3182**
| Model | PR-AUC | ROC-AUC | Log loss |
| ----- | ------ | ------- | -------- |
| CNN (Reg2M) | 0.2136 | 0.7317 | 0.2870 |
| LightGBM | 0.2088 | 0.7238 | 0.2888 |
| Blend 50/50 | 0.2245 | 0.7414 | 0.2842 |

### Log-loss gain of CNN over LightGBM (positive = First model better; 95% CI resamples game)
| Clock Filter | No. Rows | Gain | Confidence Interval |
| ------------ | -------- | ---- | ------------------- |
| All rows | 454,045 | +0.00174 | [+0.00112, +0.00228] |
| Clock <= 120s | 22,894 | -0.00064 | [-0.00332, +0.00207] |
| Clock <= 60s | 12,156 | +0.00035 | [-0.00344, +0.00401] |

### Log-loss gain of the blend over LightGBM (positive = First model better; 95% CI resamples game)
| Clock Filter | No. Rows | Gain | Confidence Interval |
| ------------ | -------- | ---- | ------------------- |
| All rows | 454,045 | +0.00457 | [+0.00426, +0.00484] |
| Clock <= 120s | 22,894 | +0.00323 | [+0.00186, +0.00463] |
| Clock <= 60s | 12,156 | +0.00378 | [+0.00185, +0.00570] |

### PR-AUC by win probability before the move
| WP Before | No. Rows | Base Rate | CNN | LightGBM |
| --------- | -------- | --------- | --- | -------- |
| 0.15-0.35 | 54,856 | 0.0804 | 0.1549 | 0.1526 |
| 0.35-0.65 | 220,345 | 0.0954 | 0.2362 | 0.2196 |
| 0.65-0.85 | 65,377 | 0.1756 | 0.2792 | 0.2707 |
| 0.85-1.00 | 113,467 | 0.0623 | 0.1676 | 0.1793 |