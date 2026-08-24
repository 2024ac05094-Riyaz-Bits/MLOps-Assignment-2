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

## Step 9 — Requirements

The project dependencies required for model training, MLflow tracking, and FastAPI inference are listed in `requirements.txt`.

The main dependencies include:

- PyTorch
- Torchvision
- Pillow
- NumPy
- MLflow
- FastAPI
- Uvicorn
- python-multipart

### Step 9 Verification

- Required dependencies listed ✓
- PyTorch import verified ✓
- MLflow import verified ✓
- FastAPI import verified ✓
- Uvicorn import verified ✓
- Requirements file verified ✓

---

## Step 10 — Docker

Docker configuration was created to containerize the FastAPI inference service.

### Dockerfile

The `Dockerfile`:

- Uses Python 3.11 slim as the base image
- Installs dependencies from `requirements.txt`
- Copies the FastAPI application
- Copies the CNN model artifact
- Exposes port `8000`
- Starts the FastAPI application using Uvicorn

### Docker Ignore

The `.dockerignore` file excludes:

- Python virtual environments
- Python cache files
- Git files
- DVC cache
- Raw and processed datasets
- MLflow local tracking files
- DVC metadata files

The trained model `artifacts/baseline_cnn.pt` is included in the Docker build because it is required for inference.

### Docker Configuration Verification

The Docker configuration was verified locally by checking:

- `Dockerfile` ✓
- `.dockerignore` ✓
- Model artifact exists ✓
- Model artifact size: approximately 51.7 MB ✓

The Docker image build and container runtime were not executed locally because Docker installation is restricted on the development office laptop.

### Step 10 Verification

- Dockerfile created ✓
- `.dockerignore` created ✓
- FastAPI container command configured ✓
- Model artifact available ✓
- Docker build configuration verified ✓
- Local Docker build — Not executed due to environment restriction

## Step 11 — Pytest

Pytest was added to validate the preprocessing pipeline and model inference components.

### Tests Implemented

The test suite verifies:

- Processed train, validation, and test directories exist
- Processed images are RGB
- Processed images are 224x224
- Baseline CNN model artifact exists
- Saved model state dictionary can be loaded successfully
- Model produces the expected binary classification output shape

### Test Execution

Run the test suite using:

\`\`\`
pytest
\`\`\`

**Test Results**

\`\`\`
5 passed in 25.11s
\`\`\`

### Step 11 Verification

- [x] Pytest test suite implemented
- [x] Preprocessing tests passed
- [x] Model artifact tests passed
- [x] Model loading test passed
- [x] Model output shape test passed
- [x] All tests passed successfully

---

## Step 12 — GitHub Actions CI

GitHub Actions CI was implemented to automatically validate the project whenever changes are pushed to the `main` branch.

### CI Pipeline

The CI workflow performs:

- Python environment setup
- Dependency installation
- Pytest execution
- Automated validation of the project

### Step 12 Verification

- [x] GitHub Actions CI workflow implemented
- [x] Dependencies installed automatically
- [x] Pytest executed through GitHub Actions
- [x] CI workflow completed successfully

---

## Step 13 — GitHub Container Registry (GHCR)

GitHub Container Registry (GHCR) was integrated to build and publish the Docker image automatically.

### GHCR Pipeline

The GitHub Actions workflow:

- Builds the Docker image
- Logs in to GitHub Container Registry
- Pushes the image to GHCR
- Creates a `latest` image tag
- Creates a commit-specific image tag using the Git SHA

### Docker Image

```text
ghcr.io/2024ac05094-riyaz-bits/mlops-assignment-2:latest
```

### Step 13 Verification

- [x] GHCR integration implemented
- [x] Docker image built successfully
- [x] Docker image pushed to GHCR
- [x] `latest` image tag created
- [x] Commit-specific image tag created

---

## Step 14 — Docker Compose

Docker Compose was added to simplify running the inference API using the published container image.

### Docker Compose Configuration

The application uses the GHCR image:

```text
ghcr.io/2024ac05094-riyaz-bits/mlops-assignment-2:latest
```

The API is exposed on:

```text
http://localhost:8000
```

### Step 14 Verification

- [x] Docker Compose configuration added
- [x] GHCR image configured
- [x] API port 8000 configured
- [x] Container restart policy configured

---

## Step 15 — CD Automation

Continuous Deployment (CD) was implemented using GitHub Actions.

The CD workflow is triggered after the Docker image build workflow completes successfully.

### CD Pipeline

The deployment workflow:

1. Checks out the repository
2. Logs in to GHCR
3. Pulls the latest Docker image
4. Starts the application using Docker Compose
5. Waits for the API to become healthy
6. Performs smoke testing
7. Displays container status
8. Stops the application after testing

### Step 15 Verification

- [x] CD workflow implemented
- [x] GHCR authentication configured
- [x] Docker image pulled successfully
- [x] Docker Compose deployment configured
- [x] Automated deployment workflow completed successfully

---

## Step 16 — Smoke Testing

Automated smoke tests were added to verify that the deployed API is running correctly.

### Smoke Tests

The CD workflow verifies:

- `/health` endpoint
- `/` root endpoint
- Successful API startup
- Docker container status

The health endpoint is checked using:

```text
GET /health
```

The root endpoint is checked using:

```text
GET /
```

### Step 16 Verification

- [x] API health check implemented
- [x] Root endpoint smoke test implemented
- [x] Docker container startup verified
- [x] CD smoke test completed successfully

---

## Step 17 — Request/Response Logging

Request and response logging was added to the FastAPI inference service to provide visibility into API activity.

The logging functionality helps track API requests and responses during inference.

### Step 17 Verification

- [x] API request logging implemented
- [x] Response logging implemented
- [x] Logging integrated with the FastAPI application
- [x] Logging changes validated through CI/CD

---

## Step 18 — Request Counters

Request counters were added to monitor API usage.

The counters provide information about the number of requests handled by the inference API.

### Monitoring

The API tracks request activity, including inference requests, to provide basic operational monitoring.

### Step 18 Verification

- [x] Request counters implemented
- [x] API request activity tracked
- [x] Changes validated through GitHub Actions
- [x] Docker image rebuilt successfully
- [x] CD deployment and smoke tests passed

---

## Current MLOps Status

The project currently includes:

- [x] Data preprocessing with DVC
- [x] Baseline CNN model training
- [x] MLflow experiment tracking
- [x] FastAPI inference API
- [x] Docker configuration
- [x] Pytest test suite
- [x] GitHub Actions CI
- [x] GitHub Container Registry (GHCR)
- [x] Docker Compose
- [x] Continuous Deployment (CD)
- [x] Automated smoke testing
- [x] Request/response logging
- [x] API request counters

---

## Step 19 — Latency Tracking

API latency tracking was implemented in the FastAPI inference service.

### Latency Monitoring

The API tracks:

- Total number of requests
- Average request latency
- Minimum request latency
- Maximum request latency
- Per-endpoint latency statistics

Each API response also includes the processing time through the
`X-Process-Time-ms` response header.

### Metrics Endpoint

Latency information can be accessed through:

```text
GET /metrics
```

Example:

```json
{
  "total_requests": 2,
  "requests_by_endpoint": {
    "GET /": 1,
    "GET /health": 1
  },
  "latency": {
    "count": 2,
    "average_ms": 2.27,
    "min_ms": 1.95,
    "max_ms": 2.58,
    "by_endpoint": {
      "GET /": {
        "count": 1,
        "average_ms": 2.58,
        "min_ms": 2.58,
        "max_ms": 2.58
      },
      "GET /health": {
        "count": 1,
        "average_ms": 1.95,
        "min_ms": 1.95,
        "max_ms": 1.95
      }
    }
  }
}
```

### Step 19 Verification

- [x] Request processing time tracked
- [x] Average latency calculated
- [x] Minimum latency calculated
- [x] Maximum latency calculated
- [x] Per-endpoint latency tracked
- [x] `/metrics` endpoint implemented
- [x] `X-Process-Time-ms` response header implemented
- [x] API tested locally
- [x] Pytest passed successfully

---

## Future MLOps Steps

  20. Model performance monitoring
  21. Final documentation
  22. Final ZIP
  23. Screen recording