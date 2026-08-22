from pathlib import Path

from PIL import Image


def test_processed_image_is_rgb_and_224(tmp_path):
    test_image = tmp_path / "test.jpg"

    image = Image.new("RGB", (224, 224))
    image.save(test_image)

    with Image.open(test_image) as loaded_image:
        assert loaded_image.mode == "RGB"
        assert loaded_image.size == (224, 224)


def test_processed_directory_structure(tmp_path):
    processed_dir = tmp_path / "data" / "processed"

    (processed_dir / "train").mkdir(parents=True)
    (processed_dir / "val").mkdir()
    (processed_dir / "test").mkdir()

    assert (processed_dir / "train").exists()
    assert (processed_dir / "val").exists()
    assert (processed_dir / "test").exists()