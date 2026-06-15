from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fire-focused YOLO dataset view using symlinks.")
    parser.add_argument("--source", default="datasets/mydata_fire", help="Original YOLO dataset root.")
    parser.add_argument("--output", default="datasets/mydata_fire_focused", help="Output dataset view root.")
    parser.add_argument("--smoke-only-keep", type=float, default=0.35, help="Fraction of smoke-only train images to keep.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def classes_in_label(path: Path) -> set[str]:
    classes: set[str] = set()
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts:
            classes.add(parts[0])
    return classes


def link_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def build_split(source: Path, output: Path, split: str, smoke_only_keep: float, rng: random.Random) -> None:
    src_images = source / split / "images"
    src_labels = source / split / "labels"
    out_images = output / split / "images"
    out_labels = output / split / "labels"

    for folder in (out_images, out_labels):
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.iterdir():
            old.unlink()

    selected: list[Path] = []
    smoke_only: list[Path] = []

    for label in sorted(src_labels.glob("*.txt")):
        classes = classes_in_label(label)
        if "0" in classes:
            selected.append(label)
        elif split == "train" and "1" in classes:
            smoke_only.append(label)
        else:
            selected.append(label)

    if split == "train" and smoke_only:
        keep_count = round(len(smoke_only) * smoke_only_keep)
        selected.extend(rng.sample(smoke_only, keep_count))

    for label in selected:
        image = src_images / f"{label.stem}.jpg"
        if not image.exists():
            image = src_images / f"{label.stem}.png"
        if not image.exists():
            continue
        link_file(image, out_images / image.name)
        link_file(label, out_labels / label.name)

    print(f"{split}: selected {len(selected)} labels")


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    rng = random.Random(args.seed)

    for split in ("train", "valid", "test"):
        build_split(source, output, split, args.smoke_only_keep, rng)

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
