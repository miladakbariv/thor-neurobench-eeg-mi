# THOR × NeuroBench 2026 - EEG Motor-Imagery Classification

**Team:** EdgeSynapse AI  
**Participant:** Milad Akbarivarmazyar  
**Affiliation:** Johannes Kepler University Linz, Austria  
**Track:** Track 1 - Absolute Accuracy  
**Task:** Binary EEG motor-imagery classification  
**Framework:** PyTorch + NeuroBench 2.3.0

## 🏆 Official Challenge Result

**1st Place - Track 1: Absolute Accuracy**

| Metric | Result |
| --- | ---: |
| Official test accuracy | **73.38%** |
| Track 1 placement | **1st** |
| Weighted operations | **14.27M** |
| Official baseline accuracy | 68.58% |
| Official baseline weighted operations | 31.09M |

The winning submission achieved **73.38% official test accuracy while using less than half the weighted operations of the official baseline**.

This repository is a portfolio-safe, reproducible engineering record of my THOR × NeuroBench Challenge 2026 work on decoding **left-hand vs right-hand motor imagery** from multi-channel EEG.

It focuses on the technical workflow: baseline design, EEG-aware neural architectures, validation discipline, NeuroBench compatibility, clean-package verification, and responsible reporting of both internal and official results.

> **Reporting note:** internal validation results and official challenge results are reported separately. The final clean-package validation accuracy was **74.13%**, while the official Track 1 test accuracy was **73.38%**.

## Problem

Each EEG trial is represented in the official workflow as:

```text
(batch, 250, 62)
       │    └── EEG channels
       └────── time steps
```

The classifier outputs two logits:

```text
(batch, 2)
```

corresponding to left-hand and right-hand motor imagery.

The project uses the THOR EEG-MI / Lee2019_MI / OpenBMI family of data exposed through the NeuroBench workflow. The working split contained **7,344 training trials** and **1,728 validation trials**.

## Engineering Progression

The project was developed baseline-first rather than jumping directly to a large neural model.

| Stage | Internal validation accuracy | Main lesson |
| --- | ---: | --- |
| Flattened EEG + linear SGD | 50.06% | Flattening destroys useful temporal/channel structure. |
| Statistical features + SGD | 60.47% | Even simple signal-aware features recover learnable structure. |
| Simple 1D CNN | 70.66% | Temporal/channel representation matters substantially. |
| Regularized 1D CNN | 71.76% | Regularization improved generalization. |
| Final clean-package validation | **74.13%** | Packaging, model selection and reproducible inference were verified end-to-end. |

Final clean-package check:

```text
1281 / 1728 = 74.131944%
```

The final package was independently reloaded and evaluated after ZIP construction so that the reported validation result reflected the artifact intended for submission rather than only an in-memory training object.

The close agreement between the internal clean-package result (**74.13%**) and the official test result (**73.38%**) also provides a useful post-hoc check on the validation and packaging process.

## Model Search

The Track 1 research pipeline explored multiple EEG-oriented candidates, including:

- Regularized temporal 1D CNN
- EEGNet-style depthwise/separable convolution
- ShallowConvNet-style architecture
- Filter-bank / log-variance models
- FBCSP-style features
- Multi-scale temporal convolution
- Temporal-shift test-time augmentation
- Validation-selected weighted ensembling

The public `src/champion_model.py` contains the portfolio-safe inference model definitions used by the champion-search workflow.

## NeuroBench Compatibility

The official challenge workflow uses the NeuroBench harness. The project therefore treated submission compatibility as an engineering requirement rather than an afterthought.

The public benchmark adapter demonstrates:

- `ThorEEGMI` validation loading
- PyTorch model wrapping via NeuroBench
- Classification accuracy evaluation
- Static model metrics such as footprint / parameter count
- Workload metrics exposed by the NeuroBench benchmark API

## Repository Structure

```text
.
├── .github/workflows/ci.yml
├── docs/
│   ├── REPRODUCIBILITY.md
│   └── RESULTS.md
├── scripts/
│   └── benchmark.py
├── src/
│   ├── __init__.py
│   └── champion_model.py
├── tests/
│   └── test_model_shapes.py
├── .gitignore
├── LICENSE
├── requirements.txt
└── requirements-dev.txt
```

## Quick Start

Reference environment: Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To run the public model smoke tests:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

To benchmark a compatible local champion checkpoint:

```bash
python scripts/benchmark.py --checkpoint path/to/best_model.pt
```

The checkpoint itself is not required to review the architecture or engineering structure of this public repository.

## Reproducibility Principles

The project follows several rules that were important for the challenge and remain useful in normal ML engineering:

1. Separate training, validation and official test evaluation roles.
2. Keep official input/output contracts explicit.
3. Use fixed seeds where deterministic comparison is required.
4. Record model and preprocessing assumptions.
5. Re-load the packaged artifact before trusting a submission metric.
6. Keep challenge credentials and private correspondence out of version control.
7. Distinguish internal validation from official challenge results.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the detailed checklist.

## Public vs Private Artifacts

This is intentionally a **portfolio-safe public repository**.

Included:

- Model architecture definitions
- NeuroBench benchmark adapter
- Internal experiment summary
- Official challenge result summary
- Reproducibility documentation
- Automated architecture smoke tests

Intentionally excluded:

- Secure submission/verification codes
- Private organizer correspondence
- Generated submission ZIPs
- Local datasets and caches
- Large/private training checkpoints

## What This Project Demonstrates

- EEG / BCI signal classification
- PyTorch model development
- Baseline-driven experimentation
- Deep-learning model comparison
- Reproducible validation
- NeuroBench integration
- Efficiency-aware benchmarking
- Submission packaging and artifact verification
- Responsible technical communication

## Challenge Context

The THOR-NeuroBench Challenge evaluates neuromorphic approaches to EEG motor-imagery decoding. Track 1 focuses on classification accuracy, while Track 2 adds compute-efficiency constraints such as synaptic operations and model footprint. The official NeuroBench harness is used for standardized evaluation.

## License

Released under the [MIT License](LICENSE).
