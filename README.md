# MLOps Assignment 2

This project implements an end-to-end MLOps pipeline for **Cats vs Dogs binary image classification** for a pet adoption platform.

## Current Status

- Step 1: Project scaffold ✓
- Step 2: Dataset preprocessing ✓
- Step 3: DVC setup and data versioning ✓
- Step 4: Git versioning ✓
- Step 5: Baseline CNN model ✓
- Step 6: Model saving ✓
- Step 7: MLflow tracking ✓

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

DVC is used to track the large dataset and preprocessing outputs, while Git is used for source-code and project configuration versioning.

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
| Epochs          | 2               |
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

| Metric              | Epoch 1  | Epoch 2  |
|----------------------|----------|----------|
| Training loss        | 0.6082   | 0.5015   |
| Training accuracy    | 66.54%   | 75.88%   |
| Validation loss      | 0.5460   | 0.4743   |
| Validation accuracy  | 72.22%   | 78.38%   |

The baseline model was successfully trained for two epochs.

### Training Command

```bash
python scripts/train.py --epochs 2 --batch-size 32
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

## Step 6 — Model Saving

The trained Baseline CNN model is saved as a PyTorch `state_dict` after training.

The model is saved to:

```text
artifacts/
└── baseline_cnn.pt
```

The saved model was verified by loading the `state_dict` into a new `BaselineCNN` instance.

### Model Verification

A dedicated verification script is provided:

```bash
python scripts/verify_model.py
```

Example output:

```text
Model loaded successfully
Model path: ...\artifacts\baseline_cnn.pt
Parameters: 12938561
```

### Step 6 Verification

- Model saved as PyTorch state_dict ✓
- Model artifact available ✓
- Model loaded successfully ✓
- Model parameters verified ✓

---

## Step 7 — MLflow Experiment Tracking

MLflow is used to track the CNN training experiment, including training configuration, metrics, and the training run.

### MLflow Configuration

**MLflow version:** 3.15.1

A local SQLite database is used as the MLflow tracking backend:

```text
mlflow.db
```

The MLflow database is excluded from Git using `.gitignore`.

The MLflow experiment name is:

```text
cats-vs-dogs-baseline
```

### Training Configuration

| Parameter      | Value           |
|-----------------|-----------------|
| Model           | BaselineCNN     |
| Epochs          | 2               |
| Batch size      | 32              |
| Learning rate   | 0.001           |
| Random seed     | 42              |
| Device          | CPU             |
| Input size      | 224x224 RGB     |

### MLflow Run

The baseline CNN training run was successfully recorded in MLflow.

- **Experiment:** `cats-vs-dogs-baseline`
- **Run ID:** `5fc917c75d064400bc1e7a05c9574e26`

### Training Metrics

The following training and validation metrics were recorded:

| Metric              | Epoch 1  | Epoch 2  |
|----------------------|----------|----------|
| Training loss        | 0.6082   | 0.5015   |
| Training accuracy    | 66.54%   | 75.88%   |
| Validation loss      | 0.5460   | 0.4743   |
| Validation accuracy  | 72.22%   | 78.38%   |

The validation accuracy improved from 72.22% to 78.38% after the second epoch.

### MLflow Verification

MLflow tracking was verified successfully:

- MLflow installed ✓
- MLflow tracking imports verified ✓
- SQLite tracking backend configured ✓
- MLflow database initialized ✓
- Experiment created ✓
- Training run recorded ✓
- Run ID generated ✓
- Training metrics recorded ✓

**Result:** Baseline CNN training was successfully tracked using MLflow. ✓

---

## Step 8 — FastAPI Inference

A FastAPI-based inference service was implemented to expose the trained Baseline CNN model through a REST API.

The API loads the trained model from:

```text
artifacts/
└── baseline_cnn.pt
```

The uploaded image is:

- Converted to RGB
- Resized to 224x224
- Converted to a PyTorch tensor
- Normalized using the same mean and standard deviation used during training

### API Endpoints

| Method | Endpoint   | Description                              |
|--------|------------|-------------------------------------------|
| GET    | `/`        | Returns basic API information             |
| GET    | `/health`  | Checks API and model health                |
| POST   | `/predict` | Predicts whether an image is a Cat or Dog  |

### Run the API

```bash
uvicorn api.main:app --reload
```

The API is available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Prediction Response

The `/predict` endpoint accepts an image file and returns the predicted class, confidence, and Dog probability.

Example Cat prediction:

```json
{
  "filename": "cat_00007.jpg",
  "prediction": "Cat",
  "confidence": 0.5425,
  "dog_probability": 0.4575
}
```

Example Dog prediction:

```json
{
  "filename": "dog_00001.jpg",
  "prediction": "Dog",
  "confidence": 0.5264,
  "dog_probability": 0.5264
}
```

### Step 8 Verification

- FastAPI installed ✓
- Uvicorn installed ✓
- API server started successfully ✓
- `/` endpoint verified ✓
- `/health` endpoint verified ✓
- `/predict` endpoint verified ✓
- OpenAPI documentation verified ✓
- Cat image prediction verified ✓
- Dog image prediction verified ✓
- Baseline CNN model loaded successfully ✓

---

## Future MLOps Steps

- `requirements.txt`
- Docker
- Pytest
- GitHub Actions CI
- GHCR
- Docker Compose
- CD automation
- Smoke testing
- Request/response logging
- Request counters
- Latency tracking
- Model performance monitoring
- Final documentation
- Final ZIP
- Screen recording