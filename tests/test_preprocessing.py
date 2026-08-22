from pathlib import Path

from PIL import Image


PROCESSED_DIR = Path("data/processed")


def test_processed_directories_exist():
    assert (PROCESSED_DIR / "train").exists()
    assert (PROCESSED_DIR / "val").exists()
    assert (PROCESSED_DIR / "test").exists()


def test_processed_image_is_rgb_and_224():
    image_files = list((PROCESSED_DIR / "test").rglob("*.jpg"))

    assert image_files, "No processed test images found."

    with Image.open(image_files[0]) as image:
        assert image.mode == "RGB"
        assert image.size == (224, 224)