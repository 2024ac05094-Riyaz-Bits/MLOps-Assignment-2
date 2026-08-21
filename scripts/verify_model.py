"""Verify that the saved CNN model can be loaded successfully."""

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import BaselineCNN


def main() -> None:
    """Load and verify the saved CNN model."""

    model_path = PROJECT_ROOT / "artifacts" / "baseline_cnn.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = BaselineCNN()

    state_dict = torch.load(
        model_path,
        map_location="cpu",
    )

    model.load_state_dict(state_dict)
    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Model loaded successfully")
    print(f"Model path: {model_path}")
    print(f"Parameters: {parameter_count}")


if __name__ == "__main__":
    main()