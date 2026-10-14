**Device:** cuda ;  **clock features:** True ;  **hybrid:** True ;  **tag:** hybrid2m
**train rows:** 2,000,000 ; **val rows:** 454,045 ; **val blunder rate:** 0.0969
**parameters:** 1,009,729 ;  **scalar inputs:** 33
| Epoch | Train Loss | Val Log Loss | Val PR-AUC | Time (s) |
| ----- | ---------- | ------------ | ---------- | -------- |
| 1 | 0.2989 | 0.2902 | 0.2074 | 27 |
| 2 | 0.2892 | 0.2892 | 0.2102 | 22 |
| 3 | 0.2872 | 0.2878 | 0.2156 | 22 |
| 4 | 0.2857 | 0.2858 | 0.2193 | 22 |
| 5 | 0.2841 | 0.2846 | 0.2228 | 22 |
| 6 | 0.2821 | 0.2839 | 0.2247 | 22 |
| 7 | 0.2795 | 0.2838 | 0.2250 | 22 |
| 8 | 0.2757 | 0.2840 | 0.2248 | 22 |
| 9 | 0.2700 | 0.2862 | 0.2208 | 22 |
| 10 | 0.2653 | 0.2885 | 0.2172 | 22 |
*early stopping*

**constant-prediction log loss: 0.3182**
| Model | PR-AUC | ROC-AUC | Log loss |
| ----- | ------ | ------- | -------- |
| CNN (Hybrid2M) | 0.2250 | 0.7427 | 0.2838 |
| LightGBM | 0.2088 | 0.7238 | 0.2888 |
| Blend 50/50 | 0.2269 | 0.7435 | 0.2834 |

### Log-loss gain of CNN over LightGBM (positive = First model better; 95% CI resamples games)
| Clock Filter | No. Rows | Gain | Confidence Interval |
| ------------ | -------- | ---- | ------------------- |
| All rows | 454,045 | +0.00493 | [+0.00439, +0.00545] |
| Clock <= 120s | 22,894 | +0.00217 | [+0.00012, +0.00440] |
| Clock <= 60s | 12,156 | +0.00251 | [-0.00048, +0.00549] |

### Log-loss gain of the blend over LightGBM (positive = First model better; 95% CI resamples games)
| Clock Filter | No. Rows | Gain | Confidence Interval |
| ------------ | -------- | ---- | ------------------- |
| All rows | 454,045 | +0.00540 | [+0.00512, +0.00566] |
| Clock <= 120s | 22,894 | +0.00376 | [+0.00264, +0.00498] |
| Clock <= 60s | 12,156 | +0.00399 | [+0.00245, +0.00564] |

### PR-AUC by win probability before the move
| WP Before | No. Rows | Base Rate | CNN | LightGBM |
| --------- | -------- | --------- | --- | -------- |
| 0.15-0.35 | 54,856 | 0.0804 | 0.1690 | 0.1526 |
| 0.35-0.65 | 220,345 | 0.0954 | 0.2400 | 0.2196 |
| 0.65-0.85 | 65,377 | 0.1756 | 0.2864 | 0.2707 |
| 0.85-1.00 | 113,467 | 0.0623 | 0.1840 | 0.1793 |