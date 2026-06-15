from __future__ import annotations

import argparse
import random
from pathlib import Path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a mixed YOLO dataset view using symlinks.")
    parser.add_argument("--mydata", default="datasets/mydata_fire_focused", help="Fire-focused project dataset root.")
    parser.add_argument(
        "--home",
        default="external_datasets/home_fire_dataset/.kagglehub_cache/datasets/pengbo00/home-fire-dataset/versions/1",
        help="Home Fire dataset root.",
    )
    parser.add_argument("--output", default="datasets/mixed_indoor_fire_70_30", help="Output dataset view root.")
    parser.add_argument("--home-train-ratio", type=float, default=0.30, help="Target Home Fire share in train split.")
    parser.add_argument("--home-val-ratio", type=float, default=0.50, help="Target Home Fire share in val split.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.iterdir():
        if old.is_file() or old.is_symlink():
            old.unlink()


def find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        image = images_dir / f"{stem}{suffix}"
        if image.exists():
            return image
    return None


def collect_items(root: Path, split: str, label_split: str | None = None) -> list[tuple[Path, Path]]:
    images_dir = root / split / "images"
    labels_dir = root / (label_split or split) / "labels"
    items: list[tuple[Path, Path]] = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = labels_dir / f"{image.stem}.txt"
        if label.exists():
            items.append((image, label))
    return items


def sample_to_ratio(
    base_count: int,
    candidates: list[tuple[Path, Path]],
    target_ratio: float,
    rng: random.Random,
) -> list[tuple[Path, Path]]:
    if target_ratio <= 0:
        return []
    if target_ratio >= 1:
        return list(candidates)
    target_count = round(base_count * target_ratio / (1 - target_ratio))
    target_count = min(target_count, len(candidates))
    return rng.sample(candidates, target_count)


def link_item(image: Path, label: Path, out_images: Path, out_labels: Path, prefix: str) -> None:
    image_dst = out_images / f"{prefix}_{image.name}"
    label_dst = out_labels / f"{prefix}_{label.name}"
    image_dst.symlink_to(image.resolve())
    label_dst.symlink_to(label.resolve())


def build_split(
    output: Path,
    split: str,
    my_items: list[tuple[Path, Path]],
    home_items: list[tuple[Path, Path]],
) -> None:
    out_images = output / split / "images"
    out_labels = output / split / "labels"
    clean_dir(out_images)
    clean_dir(out_labels)

    for image, label in my_items:
        link_item(image, label, out_images, out_labels, "my")
    for image, label in home_items:
        link_item(image, label, out_images, out_labels, "home")

    print(f"{split}: mydata={len(my_items)} home={len(home_items)} total={len(my_items) + len(home_items)}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    mydata = Path(args.mydata)
    home = Path(args.home)
    output = Path(args.output)

    my_train = collect_items(mydata, "train")
    home_train_all = collect_items(home, "train")
    home_train = sample_to_ratio(len(my_train), home_train_all, args.home_train_ratio, rng)

    my_val = collect_items(mydata, "valid")
    home_val_all = collect_items(home, "val")
    home_val = sample_to_ratio(len(my_val), home_val_all, args.home_val_ratio, rng)

    my_test = collect_items(mydata, "test")
    home_test = collect_items(home, "test")

    build_split(output, "train", my_train, home_train)
    build_split(output, "valid", my_val, home_val)
    build_split(output, "test", my_test, home_test)

    yaml_path = output / "data.yaml"
    yaml_path.write_text(
        f"""path: {output.resolve()}
train: train/images
val: valid/images
test: test/images

names:
  0: fire
  1: smoke
""",
        encoding="utf-8",
    )
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
