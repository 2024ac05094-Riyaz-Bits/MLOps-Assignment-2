# MLOps Assignment 2

This project implements an end-to-end MLOps pipeline for **Cats vs Dogs binary image classification** for a pet adoption platform.

## Step 1 — Create the Project Structure

A clean repository structure was created for the MLOps Assignment 2 project.

The project separates the major components of the machine learning and MLOps pipeline, including data processing, model training, inference, testing, monitoring, scripts, artifacts, and deployment configuration.

### Project Structure

```text
MLOps-Assignment-2/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/
│   └── models/
│
├── api/
│   └── main.py
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_inference.py
│
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── artifacts/
│   ├── baseline_cnn.pt
│   ├── model_performance.json
│   └── training_metrics.json
│
├── .github/
│   └── workflows/
│
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── requirements.txt
├── dvc.yaml
├── dvc.lock
├── .gitignore
└── README.md

```

---


## Step 2 — Dataset Preprocessing

Raw images are stored in:

```text
data/raw/
├── Cat/
└── Dog/
```

**The preprocessing pipeline:**

1. Converts images to RGB
2. Resizes images to 224x224
3. Creates deterministic 80/10/10 train/validation/test splits
4. Uses random seed 42
5. Generates one augmented copy of each training image
6. Creates dataset manifests and split summary

### Dataset Statistics

| Split            | Total  | Cats   | Dogs   |
|-------------------|--------|--------|--------|
| Raw               | 24,998 | 12,499 | 12,499 |
| Train             | 19,998 | 9,999  | 9,999  |
| Validation        | 2,498  | 1,249  | 1,249  |
| Test              | 2,502  | 1,251  | 1,251  |
| Augmented Train   | 39,996 | —      | —      |

### Run Preprocessing

```bash
python scripts/preprocess.py --raw-dir data/raw --processed-dir data/processed --seed 42
```

### Generated Structure

```text
data/processed/
├── manifests/
├── train/
├── train_augmented/
├── val/
└── test/
```

### Generated Manifests

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

The pipeline can be executed and checked using:

```bash
dvc repro
dvc status
```

After running the pipeline, `dvc.lock` is generated to record the pipeline state and data dependencies.

The DVC pipeline was successfully executed, and the data and pipeline were reported as up to date. Processed images were also verified as 224x224 RGB images.

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

The repository status, configured remote, and active branch can be checked using:

```bash
git status
git remote -v
git branch
```

The Git repository was successfully initialized, connected to GitHub, and the project files were pushed to the `main` branch.

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

| Parameter     | Value          |
|---------------|----------------|
| Model         | BaselineCNN    |
| Epochs        | 10             |
| Batch size    | 32             |
| Learning rate | 0.001          |
| Random seed   | 42             |
| Device        | CPU            |
| Input size    | 224x224 RGB    |

### Training Dataset

The model was trained using:

- Training images: 19,998
- Validation images: 2,498

### Training Results

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
|------:|-----------:|----------------:|-----------------:|----------------------:|
| 1     | 0.6082     | 66.54%          | 0.5460           | 72.22%                |
| 2     | 0.5015     | 75.88%          | 0.4743           | 78.38%                |
| 3     | 0.4216     | 80.92%          | 0.4235           | 79.94%                |
| 4     | 0.3501     | 84.84%          | 0.4300           | 81.18%                |
| 5     | 0.2704     | 88.51%          | 0.4585           | 79.18%                |
| 6     | 0.2032     | 91.85%          | 0.4736           | 81.75%                |
| 7     | 0.1468     | 94.41%          | 0.5911           | 82.07%                |
| 8     | 0.1082     | 95.86%          | 0.5756           | **82.43%**            |
| 9     | 0.0861     | 96.79%          | 0.7284           | 82.19%                |
| 10    | 0.0670     | 97.57%          | 0.8804           | 82.19%                |

The model was trained for 10 epochs. Training accuracy increased throughout the training process, while validation accuracy reached approximately **82.43% at epoch 8**.

### Training Command

```bash
python scripts/train.py --epochs 10 --batch-size 32
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

The trained model is approximately 51.7 MB and is tracked separately because of its size.

---

## Step 6 — Model Saving

The trained Baseline CNN model is saved as a PyTorch `state_dict` after training.

The model is saved to:

```text
artifacts/
└── baseline_cnn.pt
```

The saved model can be loaded into a new `BaselineCNN` instance for inference.

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

The model artifact is approximately 51.7 MB and is tracked separately from the source code.

### Model Artifact

```text
artifacts/
├── baseline_cnn.pt
└── baseline_cnn.pt.dvc
```

The `.dvc` file is used to track the model artifact with DVC.

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

| Parameter     | Value          |
|---------------|----------------|
| Model         | BaselineCNN    |
| Epochs        | 10             |
| Batch size    | 32             |
| Learning rate | 0.001          |
| Random seed   | 42             |
| Device        | CPU            |
| Input size    | 224x224 RGB    |

### MLflow Run

The Baseline CNN training run was recorded in MLflow.

- **Experiment:** `cats-vs-dogs-baseline`
- **Run ID:** `612883f2e7a641779384e508b6f5e751`

The run records the training parameters and epoch-level training and validation metrics.

### Training Metrics

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|------:|---------------:|--------------------:|------------------:|-----------------------:|
| 1     | 0.6082         | 66.54%              | 0.5460            | 72.22%                 |
| 2     | 0.5015         | 75.88%              | 0.4743            | 78.38%                 |
| 3     | 0.4216         | 80.92%              | 0.4235            | 79.94%                 |
| 4     | 0.3501         | 84.84%              | 0.4300            | 81.18%                 |
| 5     | 0.2704         | 88.51%              | 0.4585            | 79.18%                 |
| 6     | 0.2032         | 91.85%              | 0.4736            | 81.75%                 |
| 7     | 0.1468         | 94.41%              | 0.5911            | 82.07%                 |
| 8     | 0.1082         | 95.86%              | 0.5756            | 82.43%                 |
| 9     | 0.0861         | 96.79%              | 0.7284            | 82.19%                 |
| 10    | 0.0670         | 97.57%              | 0.8804            | 82.19%                 |

The model's training accuracy increased throughout the 10 epochs, reaching 97.57% at epoch 10. The highest validation accuracy was 82.43% at epoch 8.

### MLflow Artifacts

The training run also logs the following artifacts:

```text
training/
└── training_metrics.json

model/
└── baseline_cnn.pt
```

MLflow therefore provides a record of the training parameters, epoch-level metrics, and generated model artifacts.

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
|--------|-----------|-------------------------------------------|
| GET    | `/`        | Returns basic API information            |
| GET    | `/health`  | Checks API and model health              |
| POST   | `/predict` | Predicts whether an image is a Cat or Dog |

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

The `/predict` endpoint accepts an image file and returns the predicted class, confidence, and Dog probability.

### Prediction Response

Example response:

```json
{
  "filename": "cat_00007.jpg",
  "prediction": "Cat",
  "confidence": 0.5425,
  "dog_probability": 0.4575
}
```

Another example:

```json
{
  "filename": "dog_00001.jpg",
  "prediction": "Dog",
  "confidence": 0.5264,
  "dog_probability": 0.5264
}
```

The FastAPI service was tested locally using the API endpoints and the interactive Swagger documentation.

---

## Step 9 — Requirements

The project dependencies required for data preprocessing, model training, MLflow tracking, testing, and FastAPI inference are listed in `requirements.txt`.

The main dependencies include:

- PyTorch
- Torchvision
- Pillow
- NumPy
- MLflow
- FastAPI
- Uvicorn
- python-multipart
- Pytest

The `requirements.txt` file is used to install the required Python packages for the project.

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 10 — Docker

Docker configuration was created to containerize the FastAPI inference service.

### Dockerfile

The `Dockerfile`:

- Uses Python 3.11 slim as the base image
- Installs dependencies from `requirements.txt`
- Copies the FastAPI application and required project files
- Copies the trained CNN model artifact
- Exposes port `8000`
- Starts the FastAPI application using Uvicorn

### Docker Ignore

The `.dockerignore` file excludes unnecessary files and directories from the Docker build context, including:

- Python virtual environments
- Python cache files
- Git files
- DVC cache
- Raw and processed datasets
- MLflow local tracking files
- Unnecessary DVC metadata files

The trained model `artifacts/baseline_cnn.pt` is included in the Docker image because it is required for inference.

### Docker Deployment

Docker is used to package the FastAPI inference service into a container.

Docker image building and container execution were not performed locally because Docker installation is restricted on the development office laptop.

The Docker image is built and published through GitHub Actions and stored in GitHub Container Registry (GHCR). The containerized application is deployed using Docker Compose as part of the CI/CD pipeline.

### Docker Configuration

The main Docker-related files are:

```text
Dockerfile
.dockerignore
docker-compose.yml
```

---

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

```bash
pytest
```

### Test Results

```text
4 passed in 13.99s
```

The tests cover both preprocessing and inference functionality, and all tests passed successfully.

---

## Step 12 — GitHub Actions CI

GitHub Actions CI was implemented to automatically validate the project when changes are pushed to the `main` branch.

### CI Pipeline

The CI workflow performs:

- Python environment setup
- Dependency installation from `requirements.txt`
- Pytest execution
- Automated project validation

The workflow is defined in:

```text
.github/
└── workflows/
    └── ci-cd.yml
```

### CI Workflow

The GitHub Actions workflow runs the test suite automatically on the GitHub repository.

A successful CI run confirms that the project dependencies can be installed and the automated tests pass in the GitHub Actions environment.

---

## Step 13 — GitHub Container Registry (GHCR)

GitHub Container Registry (GHCR) was integrated to build and publish the Docker image automatically through GitHub Actions.

### GHCR Pipeline

The GitHub Actions workflow:

- Builds the Docker image
- Logs in to GitHub Container Registry
- Pushes the Docker image to GHCR
- Creates a `latest` image tag
- Creates a commit-specific image tag using the Git SHA

The workflow is configured in:

```text
.github/
└── workflows/
    └── ghcr.yml
```

### Docker Image

The Docker image is published to:

```text
ghcr.io/2024ac05094-riyaz-bits/mlops-assignment-2:latest
```

A commit-specific image is also created using the GitHub commit SHA:

```text
ghcr.io/2024ac05094-riyaz-bits/mlops-assignment-2:<commit-sha>
```

The GHCR workflow runs automatically when changes are pushed to the `main` branch.

---

## Step 14 — Docker Compose

Docker Compose was added to simplify running the FastAPI inference service using the published Docker image from GitHub Container Registry (GHCR).

### Docker Compose Configuration

The Compose configuration uses the published GHCR image:

```text
ghcr.io/2024ac05094-riyaz-bits/mlops-assignment-2:latest
```

The FastAPI service is exposed on port `8000`:

```text
http://localhost:8000
```

The Compose configuration also includes a restart policy so that the container can restart automatically if it stops unexpectedly.

### Docker Compose File

The configuration is defined in:

```text
deployment/
└── docker-compose.yml
```

Docker Compose provides a simple way to run the containerized FastAPI inference service with the required port and restart configuration.

---

## Step 15 — CD Automation

Continuous Deployment (CD) was implemented using GitHub Actions.

The deployment process uses the Docker image published to GitHub Container Registry (GHCR) and deploys the containerized FastAPI application using Docker Compose.

### CD Pipeline

The deployment workflow:

1. Checks out the repository
2. Logs in to GitHub Container Registry (GHCR)
3. Builds and publishes the Docker image
4. Uses the published Docker image for deployment
5. Starts the containerized FastAPI application using Docker Compose
6. Starts the application on port `8000`
7. Performs API health and smoke testing
8. Verifies that the deployed application is running successfully

### Docker Image

The published Docker image is:

```text
ghcr.io/2024ac05094-riyaz-bits/mlops-assignment-2:latest
```

The Docker image contains:

- FastAPI inference application
- Baseline CNN model
- Python dependencies
- Uvicorn server

### Docker Compose Deployment

The containerized FastAPI application is deployed using Docker Compose.

The `docker-compose.yml` configuration uses the published GHCR image and exposes the FastAPI service on port `8000`.

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

---

## Step 17 — Request/Response Logging

Request and response logging was added to the FastAPI inference service to provide visibility into API activity.

The logging functionality helps track API requests and responses during inference.

---

## Step 18 — Request Counters

Request counters were added to monitor API usage.

The counters provide information about the number of requests handled by the inference API.

### Monitoring

The API tracks request activity, including inference requests, to provide basic operational monitoring.

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

Each API response also includes the processing time through the `X-Process-Time-ms` response header.

### Metrics Endpoint

Latency information can be accessed through:

```text
GET /metrics
```

Example:

```json
{
  "total_requests": 1,
  "requests_by_endpoint": {
    "GET /": 1
  },
  "latency": {
    "count": 1,
    "average_ms": 1.58,
    "min_ms": 1.58,
    "max_ms": 1.58,
    "by_endpoint": {
      "GET /": {
        "count": 1,
        "average_ms": 1.58,
        "min_ms": 1.58,
        "max_ms": 1.58
      }
    }
  }
}
```

The `/metrics` endpoint provides basic API usage and latency information for monitoring the inference service.

---

## Step 20 — Model Performance Monitoring

Model performance evaluation was implemented to monitor the classification performance of the trained Baseline CNN model on the test dataset.

### Performance Metrics

The evaluation script calculates the following classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Number of test samples

The evaluation results are stored in:

```text
artifacts/model_performance.json
```

### Model Performance Results

The Baseline CNN model was evaluated on 2,502 test samples.

```json
{
  "test_samples": 2502,
  "accuracy": 0.8241,
  "precision": 0.8348,
  "recall": 0.8082,
  "f1_score": 0.8213
}
```

### Performance Summary

| Metric        | Score  |
|----------------|--------|
| Test Samples   | 2502   |
| Accuracy       | 82.41% |
| Precision      | 83.48% |
| Recall         | 80.82% |
| F1-score       | 82.13% |

These metrics provide a baseline for monitoring the model's classification performance and can be compared with future model versions.

### Evaluation Script

Model evaluation can be executed using:

```bash
python scripts/evaluate.py
```

The script evaluates the model on the test dataset and saves the results to:

```text
artifacts/model_performance.json
```

---
