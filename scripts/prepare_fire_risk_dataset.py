from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a single-class fire-risk YOLO dataset by mapping fire and smoke boxes to one class."
    )
    parser.add_argument(
        "--source",
        default="datasets/mixed_indoor_fire_plus_new",
        help="Source YOLO dataset with fire/smoke labels.",
    )
    parser.add_argument(
        "--output",
        default="datasets/mixed_indoor_fire_risk_plus_new",
        help="Output single-class fire-risk dataset.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy image files instead of symlinking them.",
    )
    return parser.parse_args()


def clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.iterdir():
        if old.is_file() or old.is_symlink():
            old.unlink()


def copy_or_link(source: Path, destination: Path, copy: bool) -> None:
    if copy:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def map_label(source_label: Path, output_label: Path) -> tuple[int, int]:
    input_lines = source_label.read_text(encoding="utf-8").splitlines()
    output_lines: list[str] = []
    bad_lines = 0
    for line in input_lines:
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            bad_lines += 1
            continue
        try:
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            bad_lines += 1
            continue
        x, y, w, h = coords
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            bad_lines += 1
            continue
        output_lines.append("0 " + " ".join(f"{value:.8f}" for value in coords))

    output_label.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
    return len(output_lines), bad_lines


def build_split(source: Path, output: Path, split: str, copy_images: bool) -> tuple[int, int, int, int]:
    source_images = source / split / "images"
    source_labels = source / split / "labels"
    output_images = output / split / "images"
    output_labels = output / split / "labels"
    clean_dir(output_images)
    clean_dir(output_labels)

    image_count = 0
    box_count = 0
    empty_count = 0
    bad_count = 0
    for image in sorted(source_images.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        source_label = source_labels / f"{image.stem}.txt"
        if not source_label.exists():
            continue
        output_image = output_images / image.name
        output_label = output_labels / source_label.name
        copy_or_link(image, output_image, copy_images)
        boxes, bad_lines = map_label(source_label, output_label)
        image_count += 1
        box_count += boxes
        bad_count += bad_lines
        if boxes == 0:
            empty_count += 1

    return image_count, box_count, empty_count, bad_count


def write_data_yaml(output: Path) -> None:
    (output / "data.yaml").write_text(
        f"""path: {output.resolve()}
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
    output = Path(args.output)

    for split in ("train", "valid", "test"):
        images, boxes, empty, bad = build_split(source, output, split, args.copy_images)
        print(f"{split}: images={images} boxes={boxes} empty={empty} bad_lines={bad}")
        if bad:
            raise SystemExit(f"{split}: found {bad} bad label lines")

    write_data_yaml(output)
    print(f"wrote {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
