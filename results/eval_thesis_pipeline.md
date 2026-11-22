# Evaluation: thesis_pipeline

gold labelled: 300 of 300 (pending 0, rejected 0)

## Split: test (n=200, effective n for weighted estimates ~67)

### Relevance
weighted: acc 0.648 | P 0.596 | R 0.672 | F1 0.632 | macro-F1 0.648 (95% CI 0.54-0.73) | kappa 0.298
unweighted: acc 0.655 | macro-F1 0.554 | kappa 0.117
```
pred        relevant  irrelevant
gold                            
relevant         113          43
irrelevant        26          18
```

### Topic (gold-relevant articles)
weighted: acc 0.413 | macro-F1 0.438 | kappa 0.305
```
pred                industria  agronegocio  mercado_financeiro  mercado_trabalho  servicos  none
gold                                                                                            
industria                  11            2                   4                 0         2     9
agronegocio                 6           26                   0                 0         0     2
mercado_financeiro          0            2                  13                 0         3    10
mercado_trabalho            5            3                   0                 9         6    10
servicos                    5            5                   4                 0         7    12
none                        0            0                   0                 0         0     0
```

### Sentiment, 3-class (gold-relevant articles)
weighted: acc 0.453 | macro-F1 0.422 (95% CI 0.30-0.52) | kappa 0.156 | linear-kappa 0.127
unweighted: acc 0.410 | macro-F1 0.410 | Spearman (score vs gold) 0.205 | macro-F1 excl. ambiguous 0.331
```
pred      negative  neutral  positive
gold                                 
negative        24       23         3
neutral         22       24         9
positive        17       18        16
```
per topic (weighted accuracy): agronegocio 0.66 (n=34), industria 0.42 (n=28), mercado_financeiro 0.41 (n=28), mercado_trabalho 0.43 (n=33), servicos 0.40 (n=33)

### EPU (all articles)
gold positives 18 | predicted positives 56
weighted: P 0.016 | R 0.427 | F1 0.031 (95% CI 0.01-0.06) | kappa 0.014
unweighted: P 0.196 | R 0.611 | F1 0.297
```
pred  epu  not
gold          
epu    11    7
not    45  137
```

## Split: dev (n=100, effective n for weighted estimates ~33)

### Relevance
weighted: acc 0.621 | P 0.521 | R 0.633 | F1 0.572 | macro-F1 0.616 (95% CI 0.45-0.76) | kappa 0.238
unweighted: acc 0.650 | macro-F1 0.587 | kappa 0.202
```
pred        relevant  irrelevant
gold                            
relevant          52          26
irrelevant         9          13
```

### Topic (gold-relevant articles)
weighted: acc 0.309 | macro-F1 0.343 | kappa 0.220
```
pred                industria  agronegocio  mercado_financeiro  mercado_trabalho  servicos  none
gold                                                                                            
industria                   7            1                   0                 0         0     7
agronegocio                 2            9                   1                 0         0     5
mercado_financeiro          0            4                   7                 1         2     4
mercado_trabalho            2            1                   0                 7         2     5
servicos                    3            0                   1                 0         2     5
none                        0            0                   0                 0         0     0
```

### Sentiment, 3-class (gold-relevant articles)
weighted: acc 0.480 | macro-F1 0.460 (95% CI 0.31-0.59) | kappa 0.161 | linear-kappa 0.246
unweighted: acc 0.385 | macro-F1 0.394 | Spearman (score vs gold) 0.329 | macro-F1 excl. ambiguous 0.430
```
pred      negative  neutral  positive
gold                                 
negative        10       14         0
neutral         12       11         7
positive         4       11         9
```
per topic (weighted accuracy): agronegocio 0.38 (n=17), industria 0.32 (n=15), mercado_financeiro 0.76 (n=18), mercado_trabalho 0.20 (n=17), servicos 0.80 (n=11)

### EPU (all articles)
gold positives 13 | predicted positives 22
weighted: P 0.042 | R 0.494 | F1 0.077 (95% CI 0.00-0.23) | kappa 0.045
unweighted: P 0.091 | R 0.154 | F1 0.114
```
pred  epu  not
gold          
epu     2   11
not    20   67
```

## Split: all (n=300, effective n for weighted estimates ~99)

### Relevance
weighted: acc 0.639 | P 0.570 | R 0.659 | F1 0.611 | macro-F1 0.637 (95% CI 0.53-0.71) | kappa 0.277
unweighted: acc 0.653 | macro-F1 0.567 | kappa 0.148
```
pred        relevant  irrelevant
gold                            
relevant         165          69
irrelevant        35          31
```

### Topic (gold-relevant articles)
weighted: acc 0.378 | macro-F1 0.406 | kappa 0.275
```
pred                industria  agronegocio  mercado_financeiro  mercado_trabalho  servicos  none
gold                                                                                            
industria                  18            3                   4                 0         2    16
agronegocio                 8           35                   1                 0         0     7
mercado_financeiro          0            6                  20                 1         5    14
mercado_trabalho            7            4                   0                16         8    15
servicos                    8            5                   5                 0         9    17
none                        0            0                   0                 0         0     0
```

### Sentiment, 3-class (gold-relevant articles)
weighted: acc 0.462 | macro-F1 0.437 (95% CI 0.34-0.52) | kappa 0.160 | linear-kappa 0.167
unweighted: acc 0.402 | macro-F1 0.405 | Spearman (score vs gold) 0.244 | macro-F1 excl. ambiguous 0.381
```
pred      negative  neutral  positive
gold                                 
negative        34       37         3
neutral         34       35        16
positive        21       29        25
```
per topic (weighted accuracy): agronegocio 0.56 (n=51), industria 0.39 (n=43), mercado_financeiro 0.54 (n=46), mercado_trabalho 0.36 (n=50), servicos 0.57 (n=44)

### EPU (all articles)
gold positives 31 | predicted positives 78
weighted: P 0.025 | R 0.464 | F1 0.048 (95% CI 0.01-0.11) | kappa 0.025
unweighted: P 0.167 | R 0.419 | F1 0.239
```
pred  epu  not
gold          
epu    13   18
not    65  204
```

## Strata diagnostics (all labelled)
```
                     n  gold_relevant  llm_relevant  gold_epu  llm_epu
stratum                                                               
S1_llm_relevant    150           0.95          0.62      0.01      0.0
S2_llm_epu          40           0.95          0.68      0.70      0.4
S3_llm_irrelevant   60           0.15          0.50      0.00      0.2
S4_thesis_hit       50           0.88          1.00      0.04      1.0
```
Read: S3 `gold_relevant` is the share of articles the LLM called irrelevant that you judged relevant (missed relevant). S4 `gold_epu` checks whether thesis-keyword hits the LLM did not flag are really EPU (LLM recall).
