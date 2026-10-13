| Model | precision | recall | f1 | fpr | alerts_per_day | fp_per_day | pr_auc | roc_auc | tp | fp | alerts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rule baseline | 0.0995 | 0.6632 | 0.1731 | 0.0061 | 90.4286 | 81.4286 | 0.1239 | 0.9304 | 63 | 570 | 633 |
| Isolation Forest | 0.0648 | 0.2947 | 0.1063 | 0.0043 | 61.7143 | 57.7143 | 0.0799 | 0.961 | 28 | 404 | 432 |
| Transformer | 0.0498 | 0.4526 | 0.0898 | 0.0088 | 123.2857 | 117.1429 | 0.0859 | 0.9679 | 43 | 820 | 863 |

95% bootstrap intervals (1000 resamples of test windows):

| Model | PR-AUC | precision | recall |
|---|---|---|---|
| Rule baseline | 0.080–0.203 | 0.079–0.123 | 0.567–0.760 |
| Isolation Forest | 0.042–0.153 | 0.043–0.091 | 0.204–0.387 |
| Transformer | 0.047–0.156 | 0.036–0.065 | 0.353–0.554 |
