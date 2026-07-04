from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add hard-negative empty-label images to the fire-risk YOLO dataset.")
    parser.add_argument("--source", default="datasets/mixed_indoor_fire_risk_plus_new")
    parser.add_argument("--negatives", default="datasets/openimages_hard_negatives")
    parser.add_argument("--output", default="datasets/mixed_indoor_fire_risk_plus_new_hardneg")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_split(source: Path, output: Path, split: str) -> tuple[int, int]:
    source_images = source / split / "images"
    source_labels = source / split / "labels"
    output_images = output / split / "images"
    output_labels = output / split / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    empty_count = 0
    for image in sorted(source_images.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = source_labels / f"{image.stem}.txt"
        if not label.exists():
            continue
        shutil.copy2(image, output_images / image.name)
        shutil.copy2(label, output_labels / label.name)
        image_count += 1
        if not label.read_text(encoding="utf-8").strip():
            empty_count += 1
    return image_count, empty_count


def hard_negative_images(negatives: Path) -> list[Path]:
    images_dir = negatives / "images"
    labels_dir = negatives / "labels"
    if not images_dir.exists():
        raise SystemExit(f"Missing negative image directory: {images_dir}")
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    good: list[Path] = []
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if label.exists() and not label.read_text(encoding="utf-8").strip():
            good.append(image)
    if not good:
        raise SystemExit(f"No empty-label negative images found in {negatives}")
    return good


def split_counts(total: int, train_ratio: float, valid_ratio: float) -> tuple[int, int, int]:
    train = int(total * train_ratio)
    valid = int(total * valid_ratio)
    test = total - train - valid
    return train, valid, test


def add_negatives(output: Path, images: list[Path], train_ratio: float, valid_ratio: float, seed: int) -> dict[str, int]:
    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)
    train_count, valid_count, _ = split_counts(len(shuffled), train_ratio, valid_ratio)
    split_map = {
        "train": shuffled[:train_count],
        "valid": shuffled[train_count : train_count + valid_count],
        "test": shuffled[train_count + valid_count :],
    }

    added: dict[str, int] = {}
    for split, split_images in split_map.items():
        output_images = output / split / "images"
        output_labels = output / split / "labels"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        for image in split_images:
            name = f"openimages_hardneg_{image.name}"
            shutil.copy2(image, output_images / name)
            (output_labels / f"{Path(name).stem}.txt").write_text("", encoding="utf-8")
        added[split] = len(split_images)
    return added


def write_data_yaml(output: Path) -> None:
    (output / "data.yaml").write_text(
        f"""path: {output.as_posix()}
train: train/images
val: valid/images
test: test/images

names:
  0: fire_risk
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    negatives = Path(args.negatives)
    output = Path(args.output)

    if args.train_ratio < 0 or args.valid_ratio < 0 or args.train_ratio + args.valid_ratio > 1:
        raise SystemExit("Ratios must be non-negative and train_ratio + valid_ratio must be <= 1.")
    if not (source / "data.yaml").exists():
        raise SystemExit(f"Missing source data.yaml: {source / 'data.yaml'}")

    reset_dir(output)
    for split in ("train", "valid", "test"):
        images, empty = copy_split(source, output, split)
        print(f"copied {split}: images={images} existing_empty={empty}")

    negative_images = hard_negative_images(negatives)
    added = add_negatives(output, negative_images, args.train_ratio, args.valid_ratio, args.seed)
    write_data_yaml(output)

    print(f"hard negatives available: {len(negative_images)}")
    for split in ("train", "valid", "test"):
        print(f"added {split}: {added[split]}")
    print(f"wrote {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
