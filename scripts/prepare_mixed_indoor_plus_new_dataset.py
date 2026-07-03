from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a YOLO dataset by adding the new COCO fire-only train set to mixed indoor training data."
    )
    parser.add_argument(
        "--base",
        default="datasets/mixed_indoor_fire_70_30",
        help="Existing YOLO dataset root used for train/valid/test base data.",
    )
    parser.add_argument(
        "--new",
        default="datasets/train",
        help="New dataset root containing images/, train_coco.json, and train_image.json.",
    )
    parser.add_argument(
        "--output",
        default="datasets/mixed_indoor_fire_plus_new",
        help="Output YOLO dataset root.",
    )
    parser.add_argument(
        "--copy-base",
        action="store_true",
        help="Copy base images/labels instead of symlinking them.",
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


def copy_base_split(base: Path, output: Path, split: str, copy: bool) -> int:
    src_images = base / split / "images"
    src_labels = base / split / "labels"
    out_images = output / split / "images"
    out_labels = output / split / "labels"
    clean_dir(out_images)
    clean_dir(out_labels)

    count = 0
    for image in sorted(src_images.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = src_labels / f"{image.stem}.txt"
        if not label.exists():
            continue
        image_dst = out_images / image.name
        label_dst = out_labels / label.name
        copy_or_link(image, image_dst, copy)
        copy_or_link(label, label_dst, copy)
        count += 1
    return count


def clipped_bbox_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x, y, w, h = bbox
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None
    return (
        (x1 + box_w / 2.0) / width,
        (y1 + box_h / 2.0) / height,
        box_w / width,
        box_h / height,
    )


def add_new_train_set(new_root: Path, output: Path) -> tuple[int, int, int]:
    coco = json.loads((new_root / "train_coco.json").read_text(encoding="utf-8"))
    image_labels = json.loads((new_root / "train_image.json").read_text(encoding="utf-8"))
    source_images = new_root / "images"
    out_images = output / "train" / "images"
    out_labels = output / "train" / "labels"

    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    written_images = 0
    written_boxes = 0
    empty_labels = 0
    for image_info in sorted(coco["images"], key=lambda item: item["id"]):
        source = source_images / image_info["file_name"]
        stem = f"new_{source.stem}"
        image_destination = out_images / f"{stem}.jpg"
        label_destination = out_labels / f"{stem}.txt"

        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if image.size != (image_info["width"], image_info["height"]):
                raise ValueError(
                    f"{source.name}: EXIF-normalized size {image.size} does not match COCO "
                    f"{image_info['width']}x{image_info['height']}"
                )
            image.save(image_destination, format="JPEG", quality=95)

        lines: list[str] = []
        for annotation in annotations_by_image.get(image_info["id"], []):
            if annotation["category_id"] != 0:
                raise ValueError(f"Unexpected category_id={annotation['category_id']} in {source.name}")
            yolo_box = clipped_bbox_to_yolo(
                annotation["bbox"],
                image_info["width"],
                image_info["height"],
            )
            if yolo_box is None:
                continue
            written_boxes += 1
            lines.append("0 " + " ".join(f"{value:.8f}" for value in yolo_box))

        image_level_label = image_labels.get(image_info["file_name"])
        if image_level_label == 1 and not lines:
            raise ValueError(f"{source.name}: image-level positive label has no valid boxes")
        if image_level_label == 0 and lines:
            raise ValueError(f"{source.name}: image-level negative label has detection boxes")

        label_destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if not lines:
            empty_labels += 1
        written_images += 1

    return written_images, written_boxes, empty_labels


def write_data_yaml(output: Path) -> None:
    (output / "data.yaml").write_text(
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


def main() -> None:
    args = parse_args()
    base = Path(args.base)
    new_root = Path(args.new)
    output = Path(args.output)

    train_base = copy_base_split(base, output, "train", args.copy_base)
    valid_base = copy_base_split(base, output, "valid", args.copy_base)
    test_base = copy_base_split(base, output, "test", args.copy_base)
    new_images, new_boxes, new_empty = add_new_train_set(new_root, output)
    write_data_yaml(output)

    print(f"train: base={train_base} new={new_images} total={train_base + new_images}")
    print(f"valid: base={valid_base}")
    print(f"test: base={test_base}")
    print(f"new annotations: boxes={new_boxes} empty_labels={new_empty}")
    print(f"wrote {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
