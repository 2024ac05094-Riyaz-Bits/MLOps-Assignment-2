# MLOps Assignment 2

This project implements an end-to-end MLOps pipeline for **Cats vs Dogs binary image classification** for a pet adoption platform.

## Current Status

* Step 1: Project scaffold ✓
* Step 2: Dataset preprocessing ✓
* Step 3: DVC setup and data versioning ✓

## Step 2 — Dataset Preprocessing

Raw images are stored in:

```text
data/raw/
├── Cat/
└── Dog/
```

The preprocessing pipeline:

* Converts images to RGB
* Resizes images to `224x224`
* Creates deterministic `80/10/10` train/validation/test splits
* Uses random seed `42`
* Generates one augmented copy of each training image
* Creates dataset manifests and split summary

### Dataset Statistics

| Split           |  Total |   Cats |   Dogs |
| --------------- | -----: | -----: | -----: |
| Raw             | 24,998 | 12,499 | 12,499 |
| Train           | 19,998 |  9,999 |  9,999 |
| Validation      |  2,498 |  1,249 |  1,249 |
| Test            |  2,502 |  1,251 |  1,251 |
| Augmented Train | 39,996 |      — |      — |

Run preprocessing:

```bash
python scripts/preprocess.py --raw-dir data/raw --processed-dir data/processed --seed 42
```

Generated structure:

```text
data/processed/
├── manifests/
├── train/
├── train_augmented/
├── val/
└── test/
```

Manifests:

```text
train_manifest.csv
val_manifest.csv
test_manifest.csv
split_summary.json
```

## Step 3 — DVC Data Versioning

**DVC version:** `3.67.1`

DVC is used to track the large dataset and preprocessing outputs, while Git will be used for source-code versioning.

### DVC Setup

```bash
dvc init --no-scm
dvc add data/raw
```

This creates:

```text
data/raw.dvc
.dvc/
```

### DVC Pipeline

The preprocessing stage is defined in `dvc.yaml`:

```yaml
stages:
  preprocess:
    cmd: python scripts/preprocess.py --raw-dir data/raw --processed-dir data/processed --seed 42
    deps:
      - data/raw
      - scripts/preprocess.py
    outs:
      - data/processed
```

Run and verify the pipeline:

```bash
dvc repro
dvc status
```

`dvc.lock` is generated after running the pipeline.

**Result:** Data and pipeline are up to date. ✓

## Step 3 Verification

* DVC installation ✓
* DVC initialization ✓
* Raw dataset tracking ✓
* Preprocessing pipeline ✓
* `dvc.yaml` ✓
* `dvc.lock` ✓
* Pipeline execution ✓
* DVC status verification ✓

Processed images were verified as **224x224 RGB** images.

## Future MLOps Steps

```text
4.  Git versioning
5.  Baseline CNN model
6.  Model saving
7.  MLflow tracking
8.  FastAPI inference
9.  requirements.txt
10. Docker
11. Pytest
12. GitHub Actions CI
13. GHCR
14. Docker Compose
15. CD automation
16. Smoke testing
17. Request/response logging
18. Request counters
19. Latency tracking
20. Model performance monitoring
21. Final documentation
22. Final ZIP
23. Screen recording
```

Each step will be implemented and verified before moving to the next.
