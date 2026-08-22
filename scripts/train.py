"""Train the baseline CNN model with MLflow tracking."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models import BaselineCNN


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(
        data_dir / "train",
        transform=transform,
    )

    val_dataset = datasets.ImageFolder(
        data_dir / "val",
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training or validation epoch."""

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def train_model(
    data_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    output_dir: Path,
) -> dict:
    """Train the baseline CNN and track the experiment with MLflow."""

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
    )

    model = BaselineCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = []

    print(f"Device: {device}")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    
    mlflow.set_tracking_uri(
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )
    mlflow.set_experiment("cats-vs-dogs-baseline")

    with mlflow.start_run(run_name="BaselineCNN"):

        mlflow.log_params(
            {
                "model": "BaselineCNN",
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "device": str(device),
                "input_size": "224x224",
                "loss_function": "BCEWithLogitsLoss",
                "optimizer": "Adam",
            }
        )

        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
            )

            val_loss, val_accuracy = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            epoch_result = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }

            history.append(epoch_result)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"train_accuracy={train_accuracy:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_accuracy={val_accuracy:.4f}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "baseline_cnn.pt"

        torch.save(
            model.state_dict(),
            model_path,
        )

        metrics = {
            "model": "BaselineCNN",
            "device": str(device),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "history": history,
        }

        metrics_path = output_dir / "training_metrics.json"

        with metrics_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metrics, file, indent=2)

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="training",
        )

        mlflow.log_artifact(
            str(model_path),
            artifact_path="model",
        )

        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the baseline CNN with MLflow tracking."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
    )

    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()