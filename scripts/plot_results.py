import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_METRICS_FILE = PROJECT_ROOT / "artifacts" / "training_metrics.json"
MODEL_PERFORMANCE_FILE = PROJECT_ROOT / "artifacts" / "model_performance.json"
PLOTS_DIR = PROJECT_ROOT / "artifacts" / "plots"


def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def plot_training_results(training_data):
    history = training_data["history"]

    epochs = [item["epoch"] for item in history]

    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]

    train_accuracy = [item["train_accuracy"] * 100 for item in history]
    val_accuracy = [item["val_accuracy"] * 100 for item in history]

    # Accuracy plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accuracy,
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracy,
        marker="o",
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "training_validation_accuracy.png",
        dpi=200
    )

    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_loss,
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "training_validation_loss.png",
        dpi=200
    )

    plt.close()


def plot_model_performance(performance_data):
    metrics = {
        "Accuracy": performance_data["accuracy"] * 100,
        "Precision": performance_data["precision"] * 100,
        "Recall": performance_data["recall"] * 100,
        "F1-score": performance_data["f1_score"] * 100,
    }

    plt.figure(figsize=(8, 5))

    plt.bar(
        metrics.keys(),
        metrics.values()
    )

    plt.xlabel("Metric")
    plt.ylabel("Score (%)")
    plt.title("Model Performance on Test Dataset")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "model_performance.png",
        dpi=200
    )

    plt.close()


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    training_data = load_json(TRAINING_METRICS_FILE)
    performance_data = load_json(MODEL_PERFORMANCE_FILE)

    plot_training_results(training_data)
    plot_model_performance(performance_data)

    print("Plots generated successfully.")
    print(f"Output directory: {PLOTS_DIR}")


if __name__ == "__main__":
    main()