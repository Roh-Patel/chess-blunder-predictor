train rows: 1,000,000  val rows: 454,045  val blunder rate: 0.0969
constant-prediction log loss: 0.3182
               model  n_feats  pr_auc  roc_auc  log_loss
     A: rating + ply        5  0.1382   0.6134    0.3104
        B: A + clock        8  0.1409   0.6184    0.3097
        C: A + board       28  0.2076   0.7225    0.2892
D: A + board + clock       31  0.2073   0.7224    0.2892

Log-loss gain from adding clock features (positive = clock helps; 95% CI resamples games)
A->B (no board)    all rows       n= 454,045  gain=+0.00068  CI [+0.00052, +0.00085]
A->B (no board)    clock <= 120s  n=  22,894  gain=+0.00127  CI [-0.00002, +0.00257]
A->B (no board)    clock <= 60s   n=  12,156  gain=+0.00256  CI [+0.00041, +0.00479]
C->D (with board)  all rows       n= 454,045  gain=-0.00000  CI [-0.00015, +0.00015]
C->D (with board)  clock <= 120s  n=  22,894  gain=+0.00189  CI [+0.00044, +0.00331]
C->D (with board)  clock <= 60s   n=  12,156  gain=+0.00393  CI [+0.00151, +0.00637]

Model D by win probability before the move (is it just detecting 'steep part of the curve'?)
wp_before (0.149, 0.35]  n=  54,856  base rate=0.0804  PR-AUC=0.1520  lift=1.89x
wp_before (0.35, 0.65]   n= 220,345  base rate=0.0954  PR-AUC=0.2181  lift=2.29x
wp_before (0.65, 0.85]   n=  65,377  base rate=0.1756  PR-AUC=0.2708  lift=1.54x
wp_before (0.85, 1.0]    n= 113,467  base rate=0.0623  PR-AUC=0.1765  lift=2.83x

Top features (model D, gain):
material_diff             0.189
mover_elo                 0.131
opp_elo                   0.069
ply                       0.053
my_attacked               0.052
clock_before              0.049
my_king_zone_attacked     0.048
opp_clock                 0.047
n_legal                   0.047
my_p                      0.034
total_material            0.030
opp_king_zone_attacked    0.028