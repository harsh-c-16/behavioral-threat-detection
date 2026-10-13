| Model | precision | recall | f1 | fpr | alerts_per_day | fp_per_day | pr_auc | roc_auc | tp | fp | alerts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rule baseline | 0.6579 | 0.6098 | 0.6329 | 0.0031 | 12.6667 | 4.3333 | 0.6438 | 0.9826 | 25 | 13 | 38 |
| Isolation Forest | 0.5116 | 0.5366 | 0.5238 | 0.005 | 14.3333 | 7.0 | 0.5947 | 0.9733 | 22 | 21 | 43 |
| Transformer | 0.6296 | 0.8293 | 0.7158 | 0.0048 | 18.0 | 6.6667 | 0.8474 | 0.996 | 34 | 20 | 54 |

95% bootstrap intervals (1000 resamples of test windows):

| Model | PR-AUC | precision | recall |
|---|---|---|---|
| Rule baseline | 0.501–0.778 | 0.488–0.800 | 0.454–0.758 |
| Isolation Forest | 0.439–0.734 | 0.352–0.659 | 0.378–0.682 |
| Transformer | 0.741–0.927 | 0.491–0.765 | 0.703–0.944 |
