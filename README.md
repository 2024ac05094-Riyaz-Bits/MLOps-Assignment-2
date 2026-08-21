# MLOps Assignment 2

This project implements an end-to-end MLOps pipeline for **Cats vs Dogs binary image classification** for a pet adoption platform.

## Current Status

- Step 1: Project scaffold ✓
- Step 2: Dataset preprocessing ✓
- Step 3: DVC setup and data versioning ✓
- Step 4: Git versioning ✓
- Step 5: Baseline CNN model ✓

---

## Step 2 — Dataset Preprocessing

Raw images are stored in:

```text
data/raw/
├── Cat/
└── Dog/
```

The preprocessing pipeline:

- Converts images to RGB
- Resizes images to 224x224
- Creates deterministic 80/10/10 train/validation/test splits
- Uses random seed 42
- Generates one augmented copy of each training image
- Creates dataset manifests and split summary

### Dataset Statistics

| Split            | Total  | Cats   | Dogs   |
|-------------------|--------|--------|--------|
| Raw               | 24,998 | 12,499 | 12,499 |
| Train             | 19,998 | 9,999  | 9,999  |
| Validation        | 2,498  | 1,249  | 1,249  |
| Test              | 2,502  | 1,251  | 1,251  |
| Augmented Train   | 39,996 | —      | —      |

### Run preprocessing

```bash
python scripts/preprocess.py --raw-dir data/raw --processed-dir data/processed --seed 42
```

### Generated structure

```text
data/processed/
├── manifests/
├── train/
├── train_augmented/
├── val/
└── test/
```

### Manifests

- `train_manifest.csv`
- `val_manifest.csv`
- `test_manifest.csv`
- `split_summary.json`

---

## Step 3 — DVC Data Versioning

**DVC version:** 3.67.1

DVC is used to track the large dataset and preprocessing outputs, while Git is used for source-code versioning.

### DVC Setup

```bash
dvc init --no-scm
dvc add data/raw
```

This creates:

- `data/raw.dvc`
- `.dvc/`

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

### Step 3 Verification

- DVC installation ✓
- DVC initialization ✓
- Raw dataset tracking ✓
- Preprocessing pipeline ✓
- `dvc.yaml` ✓
- `dvc.lock` ✓
- Pipeline execution ✓
- DVC status verification ✓

Processed images were verified as 224x224 RGB images.

---

## Step 4 — Git Versioning

Git is used for source-code and project configuration versioning.

### Git Setup

The project was initialized as a Git repository:

```bash
git init
```

The default branch was renamed to `main`:

```bash
git branch -M main
```

The GitHub remote repository was configured:

```bash
git remote add origin https://github.com/2024ac05094-Riyaz-Bits/MLOps-Assignment-2.git
```

The initial project files were committed and pushed to GitHub:

```bash
git add .
git commit -m "Initial MLOps Assignment 2 setup"
git push -u origin main
```

### Git Verification

```bash
git status
git remote -v
git branch
```

**Result:** Git repository initialized, connected to GitHub, and project files pushed successfully. ✓

---

## Step 5 — Baseline CNN Model

A baseline Convolutional Neural Network (CNN) was implemented using PyTorch for binary classification of Cats vs Dogs.

### Model Architecture

The `BaselineCNN` consists of:

- Convolutional layer: 3 → 32 channels
- ReLU activation
- Max pooling
- Convolutional layer: 32 → 64 channels
- ReLU activation
- Max pooling
- Convolutional layer: 64 → 128 channels
- ReLU activation
- Max pooling
- Flatten layer
- Fully connected layer: 128 → 128
- ReLU activation
- Dropout: 0.5
- Output layer: 128 → 1

The model uses `BCEWithLogitsLoss` for binary classification and the Adam optimizer.

### Training Configuration

| Parameter      | Value           |
|-----------------|-----------------|
| Model           | BaselineCNN     |
| Epochs          | 1               |
| Batch size      | 32              |
| Learning rate   | 0.001           |
| Random seed     | 42              |
| Device          | CPU             |
| Input size      | 224x224 RGB     |

### Training Dataset

The model was trained using:

- Training images: 19,998
- Validation images: 2,498

### Training Results

| Metric              | Result   |
|----------------------|----------|
| Training loss        | 0.6082   |
| Training accuracy    | 66.54%   |
| Validation loss      | 0.5460   |
| Validation accuracy  | 72.22%   |

The baseline model was successfully trained for one epoch.

### Training Command

```bash
python scripts/train.py --epochs 1 --batch-size 32
```

### Model Artifacts

The training process generates the following artifacts:

```text
artifacts/
├── baseline_cnn.pt
└── training_metrics.json
```

The trained model and training metrics are tracked using DVC:

```text
artifacts/
├── baseline_cnn.pt
├── baseline_cnn.pt.dvc
├── training_metrics.json
└── training_metrics.json.dvc
```

The trained model is approximately 51.7 MB and is therefore not stored directly in Git.

### Step 5 Verification

- CNN model implementation ✓
- PyTorch dependencies installed ✓
- Training pipeline executed ✓
- Training dataset loaded ✓
- Validation dataset loaded ✓
- Model training completed ✓
- Model artifact generated ✓
- Training metrics generated ✓
- Model artifact tracked with DVC ✓
- Training metrics tracked with DVC ✓
- DVC status verified ✓

---

## Future MLOps Steps

6. Model saving
7. MLflow tracking
8. FastAPI inference
9. `requirements.txt`
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