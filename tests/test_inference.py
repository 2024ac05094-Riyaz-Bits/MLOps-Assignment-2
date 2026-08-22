import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import BaselineCNN


def test_model_output_shape():
    model = BaselineCNN()
    model.eval()

    sample = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(sample)

    assert output.shape == (1, 1)


def test_model_state_dict_can_be_saved_and_loaded(tmp_path):
    model = BaselineCNN()

    model_path = tmp_path / "test_model.pt"

    torch.save(model.state_dict(), model_path)

    loaded_model = BaselineCNN()

    state_dict = torch.load(
        model_path,
        map_location="cpu",
    )

    loaded_model.load_state_dict(state_dict)
    loaded_model.eval()

    assert isinstance(loaded_model, BaselineCNN)