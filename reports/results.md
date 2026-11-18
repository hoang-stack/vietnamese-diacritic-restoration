# Results

## Test (3,000 held-out articles)

| Model | Word accuracy | 95% CI | Ambiguous words | Sentence accuracy | Words/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer + 4-gram LM | 99.00% | 98.95–99.05% | 98.96% | 82.52% | 5,995 |
| 4-gram LM + classifier | 98.73% | 98.67–98.79% | 98.67% | 79.24% | 5,516 |
| 4-gram LM + beam search | 98.66% | 98.60–98.72% | 98.60% | 78.37% | 6,207 |
| 4-gram LM, pruned (served) | 98.57% | 98.51–98.64% | 98.51% | 77.21% | 6,406 |
| Transformer | 98.39% | 98.33–98.46% | 98.32% | 73.29% | – |
| Context classifier | 96.88% | 96.77–96.99% | 96.73% | 58.16% | 19,968 |
| Most frequent form | 72.89% | 72.71–73.08% | 71.48% | 3.09% | 35,573 |

Paired bootstrap against **Transformer + 4-gram LM** (resampling articles):

| vs. | Δ accuracy | 95% CI | p |
| --- | ---: | ---: | ---: |
| 4-gram LM + classifier | +0.27 pp | +0.25 to +0.29 | < 0.001 |
| 4-gram LM + beam search | +0.34 pp | +0.32 to +0.36 | < 0.001 |
| 4-gram LM, pruned (served) | +0.43 pp | +0.41 to +0.45 | < 0.001 |
| Transformer | +0.61 pp | +0.58 to +0.63 | < 0.001 |
| Context classifier | +2.12 pp | +2.05 to +2.19 | < 0.001 |
| Most frequent form | +26.11 pp | +25.95 to +26.27 | < 0.001 |

## Validation: N-gram order

| Run | Accuracy | Ambiguous | Words/s | Model MB |
| --- | ---: | ---: | ---: | ---: |
| identity | 17.38% | 13.77% | 58,244 | 0 |
| mostfreq | 72.94% | 71.49% | 35,441 | 0 |
| lm2 | 97.19% | 97.05% | 15,032 | 24 |
| lm3 | 98.62% | 98.55% | 7,834 | 182 |
| lm4 | 98.72% | 98.66% | 6,075 | 616 |

## Validation: Beam width

| Run | Accuracy | Ambiguous | Words/s |
| --- | ---: | ---: | ---: |
| beam1 | 90.75% | 90.26% | 20,448 |
| beam2 | 96.41% | 96.22% | 17,559 |
| beam4 | 98.25% | 98.17% | 13,428 |
| beam8 | 98.66% | 98.60% | 9,469 |
| beam16 | 98.72% | 98.66% | 6,115 |
| beam32 | 98.73% | 98.68% | 3,637 |

## Validation: Pruning

| Run | Accuracy | Ambiguous | Words/s | Model MB |
| --- | ---: | ---: | ---: | ---: |
| prune0 | 98.72% | 98.66% | 6,115 | 616 |
| prune2 | 98.61% | 98.55% | 6,326 | 173 |
| prune3 | 98.49% | 98.42% | 6,349 | 101 |
| prune5 | 98.33% | 98.25% | 6,386 | 62 |

## Validation: LM + context classifier

| Run | Accuracy | Ambiguous | Words/s |
| --- | ---: | ---: | ---: |
| classifier | 96.91% | 96.75% | 19,821 |
| hybrid_w0.0 | 98.72% | 98.66% | 6,275 |
| hybrid_w0.1 | 98.76% | 98.71% | 5,447 |
| hybrid_w0.25 | 98.79% | 98.73% | 5,364 |
| hybrid_w0.5 | 98.77% | 98.72% | 5,323 |
| hybrid_w1.0 | 98.66% | 98.60% | 5,313 |

## Validation: Transformer (+ LM)

| Run | Accuracy | Ambiguous |
| --- | ---: | ---: |
| transformer | 98.43% | 98.36% |
| transformer+lm_w0.25 | 98.97% | 98.93% |
| transformer+lm_w0.5 | 99.03% | 98.99% |
| transformer+lm_w1.0 | 99.02% | 98.98% |
| transformer+lm_w2.0 | 98.94% | 98.90% |
| transformer+lm_w4.0 | 98.81% | 98.76% |

## Validation: learning curve

| Model | Training articles | Training tokens | Accuracy |
| --- | ---: | ---: | ---: |
| classifier | 1,000 | 598,882 | 83.58% |
| classifier | 3,000 | 1,774,671 | 87.80% |
| classifier | 10,000 | 5,854,606 | 91.74% |
| classifier | 30,000 | 17,588,047 | 94.33% |
| classifier | 99,134 | 57,941,261 | 96.23% |
| lm | 1,000 | 598,882 | 94.47% |
| lm | 3,000 | 1,774,671 | 96.28% |
| lm | 10,000 | 5,854,606 | 97.44% |
| lm | 30,000 | 17,588,047 | 98.16% |
| lm | 99,134 | 57,941,261 | 98.72% |
| mostfreq | 1,000 | 598,882 | 72.81% |
| mostfreq | 3,000 | 1,774,671 | 72.93% |
| mostfreq | 10,000 | 5,854,606 | 72.93% |
| mostfreq | 30,000 | 17,588,047 | 72.97% |
| mostfreq | 99,134 | 57,941,261 | 72.94% |

## Most frequent errors of Transformer + 4-gram LM (test)

| Correct | Predicted | Count |
| --- | --- | ---: |
| nhưng | những | 137 |
| cùng | cũng | 101 |
| cũng | cùng | 87 |
| những | nhưng | 87 |
| cô | có | 80 |
| thành | thanh | 77 |
| thanh | thành | 75 |
| còn | con | 72 |
| hằng | hàng | 60 |
| chị | chỉ | 59 |
| ba | bà | 56 |
| đ | d | 54 |
| tự | từ | 54 |
| chờ | cho | 53 |
| hưng | hùng | 48 |
| đó | do | 46 |
| con | còn | 46 |
| dễ | để | 43 |
| có | cô | 41 |
| mỗi | mọi | 41 |

## Transformer training (Kaggle)

42.5M parameters on Tesla T4, config: `{"d_model": 384, "n_layers": 6, "n_heads": 6, "d_ff": 1536, "dropout": 0.1, "max_len": 192, "tokens_per_batch": 16000, "epochs": 3, "lr": 0.0006, "warmup": 2000, "weight_decay": 0.01, "n_input": 50000, "max_candidates": 16, "seed": 0, "train_articles": null}`

| Epoch | Train loss | Val accuracy | Seconds |
| ---: | ---: | ---: | ---: |
| 1 | 0.2909 | 97.35% | 646 |
| 2 | 0.0784 | 98.20% | 651 |
| 3 | 0.0523 | 98.43% | 652 |
