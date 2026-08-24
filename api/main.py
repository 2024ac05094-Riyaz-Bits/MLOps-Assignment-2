"""FastAPI inference API for Cats vs Dogs classification."""

from __future__ import annotations

import io
import sys
from collections import Counter
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import BaselineCNN


MODEL_PATH = PROJECT_ROOT / "artifacts" / "baseline_cnn.pt"


app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="FastAPI inference service for the Baseline CNN model.",
    version="1.0.0",
)


# ---------------------------------------------------------
# Request counters
# ---------------------------------------------------------

request_counter = Counter()
counter_lock = Lock()


@app.middleware("http")
async def count_requests(request, call_next):
    """Count API requests by HTTP method and endpoint."""

    if request.url.path != "/metrics":
        endpoint = request.url.path
        method = request.method

        with counter_lock:
            request_counter["total"] += 1
            request_counter[f"{method} {endpoint}"] += 1

    response = await call_next(request)

    return response


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

# Same preprocessing used during model training.
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

model = BaselineCNN()

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu")
)

model.eval()


# ---------------------------------------------------------
# API endpoints
# ---------------------------------------------------------

@app.get("/")
def root() -> dict[str, str]:
    """Return basic API information."""

    return {
        "message": "Cats vs Dogs classifier API is running",
        "model": "BaselineCNN",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Check whether the API and model are ready."""

    return {
        "status": "healthy",
        "model": "BaselineCNN",
    }


@app.get("/metrics")
def metrics() -> dict:
    """Return API request counters."""

    with counter_lock:
        counters = dict(request_counter)

    return {
        "total_requests": counters.pop("total", 0),
        "requests_by_endpoint": counters,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Predict whether an uploaded image is a Cat or Dog."""

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image.",
        ) from exc

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        dog_probability = torch.sigmoid(output).item()

    if dog_probability >= 0.5:
        predicted_class = "Dog"
        confidence = dog_probability
    else:
        predicted_class = "Cat"
        confidence = 1.0 - dog_probability

    return {
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "dog_probability": round(dog_probability, 4),
    }