# Results

This document records the internal validation progression used during the THOR × NeuroBench EEG motor-imagery project.

These values are **internal validation metrics only**. They must not be interpreted as an official THOR/NeuroBench leaderboard score or challenge placement.

## Dataset contract

- Training trials: 7,344
- Validation trials: 1,728
- Input per trial: 250 time steps × 62 EEG channels
- Task: binary left-hand vs right-hand motor imagery

## Baseline progression

| Model / method | Internal validation accuracy | Interpretation |
| --- | ---: | --- |
| Flattened EEG + linear SGD | 50.06% | Approximately random; flattening removes useful temporal/channel structure. |
| Statistical features + SGD | 60.47% | Simple signal-aware features recover useful information. |
| Simple 1D CNN | 70.66% | Temporal convolution captures materially stronger EEG structure. |
| Regularized 1D CNN | 71.76% | Regularization improved validation generalization. |
| Final clean-package validation | **74.13%** | End-to-end packaged artifact verification. |

## Final clean-package check

```text
correct: 1281
total:   1728
accuracy: 0.74131944
```

The final package was evaluated after construction and re-loading rather than reporting only the in-memory training result.

## Champion-search design

The Track 1 search code supports several complementary candidate families:

- Regularized temporal CNN
- EEGNet-style depthwise/separable CNN
- ShallowConvNet-style network
- Filter-bank neural classifier
- FBCSP-style classifier
- Multi-scale temporal convolutional network
- Temporal-shift test-time augmentation
- Weighted validation-selected ensemble
- Optional binary decision-margin threshold selection

The purpose of the diversity was to test whether complementary inductive biases could improve validation performance while retaining the official `(batch, 250, 62) -> (batch, 2)` contract.

## Reporting rule

The project deliberately separates:

1. training-set metrics,
2. internal validation metrics,
3. clean-package validation,
4. official challenge evaluation.

Only item 3 is reported as 74.13% in this repository. No claim is made about item 4.
