"""CLI entry point for dataset preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import PreprocessConfig, preprocess_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess the Cats vs Dogs dataset into train/val/test splits."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing raw cat and dog images.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory where processed outputs will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic dataset splitting.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(224, 224),
        help="Target image size written to the processed dataset.",
    )
    parser.add_argument(
        "--augmentations-per-image",
        type=int,
        default=1,
        help="Number of deterministic augmented copies to create for each training image.",
    )
    parser.add_argument(
        "--disable-train-augmentation",
        action="store_true",
        help="Disable creation of the train_augmented dataset.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = PreprocessConfig(
        raw_dir=args.raw_dir.resolve(),
        processed_dir=args.processed_dir.resolve(),
        image_size=(args.image_size[0], args.image_size[1]),
        seed=args.seed,
        augment_train=not args.disable_train_augmentation,
        augmentations_per_image=args.augmentations_per_image,
    )
    summary = preprocess_dataset(config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
