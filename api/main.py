"""FastAPI inference API for Cats vs Dogs classification."""

from __future__ import annotations

import io
import sys
import time
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
# Request counters and latency tracking
# ---------------------------------------------------------

request_counter = Counter()

latency_stats = {
    "count": 0,
    "total_ms": 0.0,
    "min_ms": None,
    "max_ms": None,
}

endpoint_latency = {}

counter_lock = Lock()


@app.middleware("http")
async def monitor_requests(request, call_next):
    """Track request counts and API response latency."""

    start_time = time.perf_counter()

    response = None

    try:
        response = await call_next(request)
        return response

    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        endpoint = request.url.path
        method = request.method

        # Do not count the monitoring endpoint itself.
        if endpoint != "/metrics":
            with counter_lock:
                # Request counters
                request_counter["total"] += 1
                request_counter[f"{method} {endpoint}"] += 1

                # Overall latency statistics
                latency_stats["count"] += 1
                latency_stats["total_ms"] += elapsed_ms

                if (
                    latency_stats["min_ms"] is None
                    or elapsed_ms < latency_stats["min_ms"]
                ):
                    latency_stats["min_ms"] = elapsed_ms

                if (
                    latency_stats["max_ms"] is None
                    or elapsed_ms > latency_stats["max_ms"]
                ):
                    latency_stats["max_ms"] = elapsed_ms

                # Per-endpoint latency statistics
                endpoint_key = f"{method} {endpoint}"

                if endpoint_key not in endpoint_latency:
                    endpoint_latency[endpoint_key] = {
                        "count": 0,
                        "total_ms": 0.0,
                        "min_ms": None,
                        "max_ms": None,
                    }

                endpoint_stats = endpoint_latency[endpoint_key]

                endpoint_stats["count"] += 1
                endpoint_stats["total_ms"] += elapsed_ms

                if (
                    endpoint_stats["min_ms"] is None
                    or elapsed_ms < endpoint_stats["min_ms"]
                ):
                    endpoint_stats["min_ms"] = elapsed_ms

                if (
                    endpoint_stats["max_ms"] is None
                    or elapsed_ms > endpoint_stats["max_ms"]
                ):
                    endpoint_stats["max_ms"] = elapsed_ms

        if response is not None:
            response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"


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
    """Return API request counters and latency statistics."""

    with counter_lock:
        counters = dict(request_counter)
        overall_latency = dict(latency_stats)
        endpoint_latency_stats = {
            endpoint: dict(stats)
            for endpoint, stats in endpoint_latency.items()
        }

    request_count = overall_latency["count"]

    if request_count > 0:
        average_ms = (
            overall_latency["total_ms"] / request_count
        )
    else:
        average_ms = 0.0

    endpoint_metrics = {}

    for endpoint, stats in endpoint_latency_stats.items():
        if stats["count"] > 0:
            endpoint_average_ms = (
                stats["total_ms"] / stats["count"]
            )
        else:
            endpoint_average_ms = 0.0

        endpoint_metrics[endpoint] = {
            "count": stats["count"],
            "average_ms": round(endpoint_average_ms, 2),
            "min_ms": round(stats["min_ms"], 2)
            if stats["min_ms"] is not None
            else 0.0,
            "max_ms": round(stats["max_ms"], 2)
            if stats["max_ms"] is not None
            else 0.0,
        }

    return {
        "total_requests": counters.pop("total", 0),
        "requests_by_endpoint": counters,
        "latency": {
            "count": request_count,
            "average_ms": round(average_ms, 2),
            "min_ms": round(overall_latency["min_ms"], 2)
            if overall_latency["min_ms"] is not None
            else 0.0,
            "max_ms": round(overall_latency["max_ms"], 2)
            if overall_latency["max_ms"] is not None
            else 0.0,
            "by_endpoint": endpoint_metrics,
        },
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