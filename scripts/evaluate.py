"""Evaluate the trained Cats vs Dogs model on the test dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import BaselineCNN


MODEL_PATH = PROJECT_ROOT / "artifacts" / "baseline_cnn.pt"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "model_performance.json"


# Same preprocessing used during training and inference.
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


def evaluate_model() -> dict[str, float | int]:
    """Evaluate the trained model on the processed test dataset."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATA_PATH}"
        )

    dataset = datasets.ImageFolder(
        TEST_DATA_PATH,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
    )

    model = BaselineCNN()

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)

            probabilities = torch.sigmoid(outputs).view(-1)

            predictions = (probabilities >= 0.5).long()

            all_predictions.extend(
                predictions.tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    metrics = {
        "test_samples": len(dataset),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    return metrics


if __name__ == "__main__":
    metrics = evaluate_model()

    print("Model Performance:")
    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print(
        f"\nMetrics saved to: {OUTPUT_PATH}"
    )