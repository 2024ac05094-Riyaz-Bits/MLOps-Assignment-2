"""Dataset preprocessing utilities for the Cats vs Dogs project."""

from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageEnhance, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
EXPECTED_LABELS = ("cat", "dog")
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class DatasetItem:
    """Represents one raw image and its inferred class label."""

    source_path: Path
    label: str
    index: int

    @property
    def stem(self) -> str:
        return f"{self.label}_{self.index:05d}"


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Configuration for dataset preprocessing."""

    raw_dir: Path
    processed_dir: Path
    image_size: tuple[int, int] = (224, 224)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    augment_train: bool = True
    augmentations_per_image: int = 1


def preprocess_dataset(config: PreprocessConfig) -> dict:
    """Preprocess the raw dataset and return a machine-readable summary."""

    validate_config(config)
    class_dirs = resolve_class_directories(config.raw_dir)
    items = discover_images(class_dirs)
    if not items:
        raise FileNotFoundError(
            f"No .jpg/.jpeg/.png/.bmp files were found under '{config.raw_dir}'. "
            "Place the Kaggle dataset inside data/raw/cat and data/raw/dog (case-insensitive folder names are allowed)."
        )

    split_mapping = split_dataset(items, config)
    reset_output_directories(config.processed_dir)
    prepare_output_directories(config.processed_dir, config.augment_train)
    write_processed_images(split_mapping, config)
    write_manifests(split_mapping, config, class_dirs)
    return build_summary(split_mapping, config, class_dirs)


def validate_config(config: PreprocessConfig) -> None:
    ratio_sum = config.train_ratio + config.val_ratio + config.test_ratio
    if round(ratio_sum, 6) != 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if config.augmentations_per_image < 0:
        raise ValueError("augmentations_per_image must be zero or greater")
    if not config.raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {config.raw_dir}")


def resolve_class_directories(raw_dir: Path) -> dict[str, Path]:
    """Resolve the expected cat and dog class folders under data/raw."""

    directories = {child.name.lower(): child for child in raw_dir.iterdir() if child.is_dir()}
    missing = [label for label in EXPECTED_LABELS if label not in directories]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required class folder(s): {missing_text}. "
            "Expected data/raw/cat and data/raw/dog (case-insensitive names are accepted)."
        )

    extras = sorted(name for name in directories if name not in EXPECTED_LABELS)
    if extras:
        extras_text = ", ".join(extras)
        raise ValueError(
            f"Unexpected class folder(s) found in raw data: {extras_text}. "
            "Keep only cat and dog folders under data/raw for this assignment."
        )

    return {label: directories[label] for label in EXPECTED_LABELS}


def discover_images(class_dirs: dict[str, Path]) -> list[DatasetItem]:
    """Find supported images in the cat and dog folders."""

    discovered: list[DatasetItem] = []
    for label, class_dir in class_dirs.items():
        label_index = 0
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_index += 1
            discovered.append(DatasetItem(source_path=path, label=label, index=label_index))

    return discovered


def split_dataset(items: Iterable[DatasetItem], config: PreprocessConfig) -> dict[str, list[DatasetItem]]:
    """Create deterministic train/val/test splits while preserving class balance."""

    rng = random.Random(config.seed)
    grouped = {label: [] for label in EXPECTED_LABELS}
    for item in items:
        grouped[item.label].append(item)

    for group_items in grouped.values():
        rng.shuffle(group_items)

    split_mapping = {split_name: [] for split_name in SPLIT_NAMES}
    for group_items in grouped.values():
        counts = compute_split_counts(
            len(group_items),
            config.train_ratio,
            config.val_ratio,
            config.test_ratio,
        )
        start = 0
        for split_name, count in zip(SPLIT_NAMES, counts):
            split_mapping[split_name].extend(group_items[start : start + count])
            start += count

    for split_name in SPLIT_NAMES:
        split_mapping[split_name].sort(key=lambda item: (item.label, item.index))

    return split_mapping


def compute_split_counts(
    total_count: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    """Convert ratios into integer counts while keeping the total exact."""

    if total_count == 0:
        return (0, 0, 0)

    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count

    if total_count >= 3:
        if val_count == 0:
            val_count = 1
            train_count -= 1
        if test_count == 0:
            test_count = 1
            train_count -= 1

    if train_count < 0:
        raise ValueError(
            "Not enough images per class to create train/val/test splits. "
            "Provide at least three images per class."
        )

    return (train_count, val_count, test_count)


def reset_output_directories(processed_dir: Path) -> None:
    """Remove previous generated outputs while preserving the processed root."""

    for child_name in ("train", "val", "test", "train_augmented", "manifests"):
        child_path = processed_dir / child_name
        if child_path.exists():
            shutil.rmtree(child_path)


def prepare_output_directories(processed_dir: Path, augment_train: bool) -> None:
    """Create the directory structure used by the processed dataset."""

    for split_name in SPLIT_NAMES:
        for label in EXPECTED_LABELS:
            (processed_dir / split_name / label).mkdir(parents=True, exist_ok=True)

    if augment_train:
        for label in EXPECTED_LABELS:
            (processed_dir / "train_augmented" / label).mkdir(parents=True, exist_ok=True)

    (processed_dir / "manifests").mkdir(parents=True, exist_ok=True)


def write_processed_images(split_mapping: dict[str, list[DatasetItem]], config: PreprocessConfig) -> None:
    """Convert images to RGB, resize them, and write split outputs."""

    for split_name, items in split_mapping.items():
        for item in items:
            processed_image = load_and_resize(item.source_path, config.image_size)
            destination = config.processed_dir / split_name / item.label / f"{item.stem}.jpg"
            processed_image.save(destination, format="JPEG", quality=95)

            if split_name == "train" and config.augment_train:
                augmented_dir = config.processed_dir / "train_augmented" / item.label
                processed_image.save(
                    augmented_dir / f"{item.stem}.jpg",
                    format="JPEG",
                    quality=95,
                )
                for augmentation_index in range(config.augmentations_per_image):
                    augmented_image = apply_augmentation(processed_image, augmentation_index)
                    augmented_image.save(
                        augmented_dir / f"{item.stem}_aug{augmentation_index + 1}.jpg",
                        format="JPEG",
                        quality=95,
                    )


def load_and_resize(image_path: Path, image_size: tuple[int, int]) -> Image.Image:
    """Load an image, convert it to RGB, and resize it to the target size."""

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        return ImageOps.fit(
            rgb_image,
            image_size,
            method=Image.Resampling.BILINEAR,
        )


def apply_augmentation(image: Image.Image, augmentation_index: int) -> Image.Image:
    """Create a deterministic augmented variant of a training image."""

    variants = (
        lambda img: ImageOps.mirror(img),
        lambda img: img.rotate(10, resample=Image.Resampling.BILINEAR),
        lambda img: ImageEnhance.Brightness(img).enhance(1.15),
        lambda img: ImageEnhance.Contrast(img).enhance(1.2),
    )
    transform = variants[augmentation_index % len(variants)]
    return transform(image)


def write_manifests(
    split_mapping: dict[str, list[DatasetItem]],
    config: PreprocessConfig,
    class_dirs: dict[str, Path],
) -> None:
    """Write CSV manifests and a JSON summary for easy inspection."""

    manifests_dir = config.processed_dir / "manifests"
    summary = build_summary(split_mapping, config, class_dirs)

    for split_name, items in split_mapping.items():
        manifest_path = manifests_dir / f"{split_name}_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["split", "label", "source_path", "processed_path"],
            )
            writer.writeheader()
            for item in items:
                writer.writerow(
                    {
                        "split": split_name,
                        "label": item.label,
                        "source_path": str(item.source_path),
                        "processed_path": str(
                            config.processed_dir / split_name / item.label / f"{item.stem}.jpg"
                        ),
                    }
                )

    summary_path = manifests_dir / "split_summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)


def build_summary(
    split_mapping: dict[str, list[DatasetItem]],
    config: PreprocessConfig,
    class_dirs: dict[str, Path],
) -> dict:
    """Build a JSON-friendly summary of the preprocessing output."""

    summary = {
        "raw_dir": str(config.raw_dir),
        "raw_class_dirs": {label: str(path) for label, path in class_dirs.items()},
        "processed_dir": str(config.processed_dir),
        "image_size": list(config.image_size),
        "seed": config.seed,
        "expected_structure": ["data/raw/cat", "data/raw/dog"],
        "splits": {},
        "train_augmentation": {
            "enabled": config.augment_train,
            "augmentations_per_image": config.augmentations_per_image,
        },
    }

    for split_name, items in split_mapping.items():
        label_counts = {label: 0 for label in EXPECTED_LABELS}
        for item in items:
            label_counts[item.label] += 1

        summary["splits"][split_name] = {
            "total": len(items),
            "class_counts": label_counts,
        }

    train_total = summary["splits"]["train"]["total"]
    summary["train_augmented_total"] = (
        train_total * (config.augmentations_per_image + 1) if config.augment_train else 0
    )
    return summary
