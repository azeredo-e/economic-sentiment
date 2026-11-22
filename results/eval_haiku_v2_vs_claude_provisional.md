# Evaluation: haiku_v2_vs_claude_provisional

gold labelled: 300 of 300 (pending 0, rejected 0)

## Split: test (n=200, effective n for weighted estimates ~67)

### Relevance
weighted: acc 0.888 | P 0.960 | R 0.784 | F1 0.863 | macro-F1 0.884 (95% CI 0.81-0.95) | kappa 0.770
unweighted: acc 0.885 | macro-F1 0.846 | kappa 0.693
```
pred        relevant  irrelevant
gold                            
relevant         139          17
irrelevant         6          38
```

### Topic (gold-relevant articles)
weighted: acc 0.724 | macro-F1 0.811 | kappa 0.658
```
pred                industria  agronegocio  mercado_financeiro  mercado_trabalho  servicos  none
gold                                                                                            
industria                  22            1                   1                 0         2     2
agronegocio                 0           31                   0                 0         1     2
mercado_financeiro          0            0                  25                 0         0     3
mercado_trabalho            2            0                   0                24         0     7
servicos                    0            0                   1                 1        28     3
none                        0            0                   0                 0         0     0
```

### Sentiment, 3-class (gold-relevant articles)
weighted: acc 0.754 | macro-F1 0.767 (95% CI 0.65-0.87) | kappa 0.630 | linear-kappa 0.715
unweighted: acc 0.827 | macro-F1 0.825 | Spearman (score vs gold) 0.883 | macro-F1 excl. ambiguous 1.000
```
pred      negative  neutral  positive
gold                                 
negative        46        4         0
neutral          5       35        15
positive         0        3        48
```
per topic (weighted accuracy): agronegocio 0.86 (n=34), industria 0.79 (n=28), mercado_financeiro 0.81 (n=28), mercado_trabalho 0.68 (n=33), servicos 0.74 (n=33)

### EPU (all articles)
gold positives 18 | predicted positives 16
weighted: P 0.807 | R 0.668 | F1 0.731 (95% CI 0.41-0.88) | kappa 0.729
unweighted: P 0.688 | R 0.611 | F1 0.647
```
pred  epu  not
gold          
epu    11    7
not     5  177
```

## Split: dev (n=100, effective n for weighted estimates ~33)

### Relevance
weighted: acc 0.967 | P 0.986 | R 0.932 | F1 0.958 | macro-F1 0.966 (95% CI 0.93-0.99) | kappa 0.932
unweighted: acc 0.920 | macro-F1 0.890 | kappa 0.781
```
pred        relevant  irrelevant
gold                            
relevant          72           6
irrelevant         2          20
```

### Topic (gold-relevant articles)
weighted: acc 0.791 | macro-F1 0.805 | kappa 0.732
```
pred                industria  agronegocio  mercado_financeiro  mercado_trabalho  servicos  none
gold                                                                                            
industria                  12            0                   2                 0         0     1
agronegocio                 1           15                   0                 0         0     1
mercado_financeiro          0            0                  17                 0         0     1
mercado_trabalho            1            0                   0                15         0     1
servicos                    0            0                   0                 2         7     2
none                        0            0                   0                 0         0     0
```

### Sentiment, 3-class (gold-relevant articles)
weighted: acc 0.702 | macro-F1 0.715 (95% CI 0.55-0.89) | kappa 0.558 | linear-kappa 0.653
unweighted: acc 0.821 | macro-F1 0.820 | Spearman (score vs gold) 0.902 | macro-F1 excl. ambiguous 0.809
```
pred      negative  neutral  positive
gold                                 
negative        22        2         0
neutral          5       19         6
positive         0        1        23
```
per topic (weighted accuracy): agronegocio 0.71 (n=17), industria 0.72 (n=15), mercado_financeiro 1.00 (n=18), mercado_trabalho 0.65 (n=17), servicos 0.47 (n=11)

### EPU (all articles)
gold positives 13 | predicted positives 12
weighted: P 1.000 | R 0.691 | F1 0.818 (95% CI 0.50-1.00) | kappa 0.815
unweighted: P 1.000 | R 0.923 | F1 0.960
```
pred  epu  not
gold          
epu    12    1
not     0   87
```

## Split: all (n=300, effective n for weighted estimates ~99)

### Relevance
weighted: acc 0.917 | P 0.970 | R 0.834 | F1 0.897 | macro-F1 0.914 (95% CI 0.86-0.95) | kappa 0.828
unweighted: acc 0.897 | macro-F1 0.860 | kappa 0.722
```
pred        relevant  irrelevant
gold                            
relevant         211          23
irrelevant         8          58
```

### Topic (gold-relevant articles)
weighted: acc 0.747 | macro-F1 0.810 | kappa 0.682
```
pred                industria  agronegocio  mercado_financeiro  mercado_trabalho  servicos  none
gold                                                                                            
industria                  34            1                   3                 0         2     3
agronegocio                 1           46                   0                 0         1     3
mercado_financeiro          0            0                  42                 0         0     4
mercado_trabalho            3            0                   0                39         0     8
servicos                    0            0                   1                 3        35     5
none                        0            0                   0                 0         0     0
```

### Sentiment, 3-class (gold-relevant articles)
weighted: acc 0.736 | macro-F1 0.750 (95% CI 0.67-0.83) | kappa 0.605 | linear-kappa 0.694
unweighted: acc 0.825 | macro-F1 0.823 | Spearman (score vs gold) 0.890 | macro-F1 excl. ambiguous 0.956
```
pred      negative  neutral  positive
gold                                 
negative        68        6         0
neutral         10       54        21
positive         0        4        71
```
per topic (weighted accuracy): agronegocio 0.80 (n=51), industria 0.77 (n=43), mercado_financeiro 0.88 (n=46), mercado_trabalho 0.67 (n=50), servicos 0.62 (n=44)

### EPU (all articles)
gold positives 31 | predicted positives 28
weighted: P 0.906 | R 0.681 | F1 0.777 (95% CI 0.55-0.92) | kappa 0.775
unweighted: P 0.821 | R 0.742 | F1 0.780
```
pred  epu  not
gold          
epu    23    8
not     5  264
```

## Strata diagnostics (all labelled)
```
                     n  gold_relevant  llm_relevant  gold_epu  llm_epu
stratum                                                               
S1_llm_relevant    150           0.95          0.95      0.01     0.01
S2_llm_epu          40           0.95          0.88      0.70     0.62
S3_llm_irrelevant   60           0.15          0.03      0.00     0.00
S4_thesis_hit       50           0.88          0.80      0.04     0.04
```
Read: S3 `gold_relevant` is the share of articles the LLM called irrelevant that you judged relevant (missed relevant). S4 `gold_epu` checks whether thesis-keyword hits the LLM did not flag are really EPU (LLM recall).
