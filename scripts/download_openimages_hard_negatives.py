from __future__ import annotations

import argparse
import csv
import random
import shutil
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


CLASS_DESCRIPTION_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
BBOX_URLS = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Open Images hard-negative images and write empty YOLO labels."
    )
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test", "all"))
    parser.add_argument(
        "--classes",
        default="Traffic light,Lamp,Candle,Torch,Street light",
        help="Comma-separated Open Images class names to use as hard negatives.",
    )
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="datasets/openimages_hard_negatives")
    parser.add_argument("--cache", default="external_datasets/openimages_cache")
    parser.add_argument("--num-candidates", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as file:
        shutil.copyfileobj(response, file)


def load_class_ids(class_file: Path, class_names: list[str]) -> dict[str, str]:
    wanted = {name.strip().lower(): name.strip() for name in class_names if name.strip()}
    matches: dict[str, str] = {}
    with class_file.open(newline="", encoding="utf-8") as file:
        for label_id, display_name in csv.reader(file):
            key = display_name.strip().lower()
            if key in wanted:
                matches[wanted[key]] = label_id
    missing = [name for name in class_names if name.strip() and name.strip() not in matches]
    if missing:
        print("classes not found:", ", ".join(missing))
    if not matches:
        raise SystemExit("No requested classes were found in Open Images boxable class names.")
    print("matched classes:")
    for name, label_id in matches.items():
        print(f"  {name}: {label_id}")
    return matches


def collect_image_ids(bbox_file: Path, label_ids: set[str], split: str) -> list[tuple[str, str]]:
    image_ids: set[str] = set()
    with bbox_file.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["LabelName"] in label_ids:
                image_ids.add(row["ImageID"])
    image_id_list = [(split, image_id) for image_id in sorted(image_ids)]
    print(f"{split} candidate images: {len(image_id_list)}")
    return image_id_list


def download_image(split: str, image_id: str, destination: Path) -> bool:
    url = IMAGE_URL.format(split=split, image_id=image_id)
    try:
        with urllib.request.urlopen(url, timeout=30) as response, destination.open("wb") as file:
            shutil.copyfileobj(response, file)
        return True
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"failed {split}/{image_id}: {exc}")
        if destination.exists():
            destination.unlink()
        return False


def main() -> None:
    args = parse_args()
    class_names = [name.strip() for name in args.classes.split(",") if name.strip()]
    cache = Path(args.cache)
    class_file = cache / "oidv7-class-descriptions-boxable.csv"

    download_file(CLASS_DESCRIPTION_URL, class_file)
    class_ids = load_class_ids(class_file, class_names)

    splits = ("validation", "test") if args.split == "all" else (args.split,)
    candidates: list[tuple[str, str]] = []
    for split in splits:
        bbox_file = cache / Path(BBOX_URLS[split]).name
        download_file(BBOX_URLS[split], bbox_file)
        candidates.extend(collect_image_ids(bbox_file, set(class_ids.values()), split))

    random.Random(args.seed).shuffle(candidates)
    if args.num_candidates > 0:
        candidates = candidates[: args.num_candidates]
    print(f"candidate images total: {len(candidates)}")
    candidates = candidates[: args.max_images]

    output = Path(args.output)
    images_dir = output / "images"
    labels_dir = output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    existing = len([path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if existing:
        print(f"found existing images: {existing}")

    def fetch(candidate: tuple[str, str]) -> bool:
        split, image_id = candidate
        image_path = images_dir / f"openimages_{split}_{image_id}.jpg"
        label_path = labels_dir / f"{image_path.stem}.txt"
        if image_path.exists():
            label_path.write_text("", encoding="utf-8")
            return True
        if download_image(split, image_id, image_path):
            label_path.write_text("", encoding="utf-8")
            return True
        return False

    downloaded = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch, candidate) for candidate in candidates]
        for future in as_completed(futures):
            if future.result():
                downloaded += 1
                if downloaded % 50 == 0:
                    print(f"ready {downloaded}/{min(args.max_images, len(candidates))}")
            if downloaded >= args.max_images:
                break

    (output / "data.yaml").write_text(
        f"""path: {output.as_posix()}
train: images
val: images
test: images

names:
  0: fire_risk
""",
        encoding="utf-8",
    )
    print(f"wrote {downloaded} hard-negative images to {output}")


if __name__ == "__main__":
    main()
