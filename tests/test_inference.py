import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models import BaselineCNN


MODEL_PATH = PROJECT_ROOT / "artifacts" / "baseline_cnn.pt"


def test_model_artifact_exists():
    assert MODEL_PATH.exists()


def test_model_can_be_loaded():
    model = BaselineCNN()

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model.load_state_dict(state_dict)
    model.eval()

    assert isinstance(model, BaselineCNN)


def test_model_output_shape():
    model = BaselineCNN()
    model.eval()

    sample = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(sample)

    assert output.shape == (1, 1)