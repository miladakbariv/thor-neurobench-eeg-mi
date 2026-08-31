# Reproducibility and Submission Hygiene

The THOR × NeuroBench work was treated as a research-engineering project, not only a model-training exercise. This checklist captures the rules used to reduce leakage, packaging mistakes and accidental disclosure.

## Environment

Reference runtime:

```text
Python 3.11
PyTorch
NeuroBench 2.3.0
NumPy
SciPy
scikit-learn
```

Create a clean environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data contract

The public code assumes the official model input contract:

```text
(batch, time, channels) = (batch, 250, 62)
```

Model output:

```text
(batch, 2)
```

The public repository does not redistribute local dataset caches.

## Experiment discipline

For each serious experiment:

1. Record the model/configuration and purpose.
2. Fix or explicitly record random seeds.
3. Keep training and validation roles separate.
4. Never select hyperparameters from an unseen challenge test score.
5. Record validation accuracy and relevant diagnostics.
6. Save the candidate only when it improves the chosen selection criterion.
7. Re-run inference from the saved artifact before trusting the result.

## Clean-package verification

A submission artifact should be treated as a separate object from the training workspace.

Recommended sequence:

```text
train/select model
        ↓
save checkpoint
        ↓
build clean package
        ↓
unpack into fresh directory
        ↓
load model from packaged files
        ↓
run validation inference again
        ↓
compare metric with pre-package result
```

The final internal package used in this project passed this re-load/evaluation step at **1281 / 1728 = 74.13%** validation accuracy.

## Security / privacy checklist

Never commit:

- Secure challenge submission or verification codes
- Private organizer emails
- Personal credentials or API keys
- Private screenshots
- Generated submission ZIPs unless intentionally approved for public release
- Local raw/cache datasets

The `.gitignore` contains defensive patterns for these artifact classes, but manual review before every push is still required.

## Before release

```text
[ ] README clearly distinguishes internal validation from official results
[ ] requirements.txt exists
[ ] tests pass in a clean environment
[ ] model input/output contract is documented
[ ] no absolute local paths remain
[ ] no credentials or secure challenge codes exist in tracked files
[ ] no private correspondence is included
[ ] generated artifacts/checkpoints are reviewed before publication
```
