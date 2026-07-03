from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import yaml
from PIL import Image, ImageDraw
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fire-risk detection across confidence thresholds.")
    parser.add_argument("--model", required=True, help="YOLO weight path, for example runs/train/.../weights/best.pt")
    parser.add_argument("--data", default="datasets/mixed_indoor_fire_risk_plus_new/data.yaml")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold used to match predictions to labels.")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", default="runs/threshold_analysis/fire_risk")
    parser.add_argument("--export-threshold", type=float, default=0.10)
    parser.add_argument("--max-export", type=int, default=80)
    return parser.parse_args()


def load_dataset(data_yaml: Path, split: str) -> tuple[Path, Path]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(data.get("path", ""))
    if not dataset_root.exists():
        dataset_root = data_yaml.parent

    split_key = "val" if split == "val" else split
    images = Path(data[split_key])
    if not images.is_absolute():
        images = dataset_root / images
    labels = Path(str(images).replace("/images", "/labels"))
    if not images.exists():
        raise SystemExit(f"Missing images directory: {images}")
    if not labels.exists():
        raise SystemExit(f"Missing labels directory: {labels}")
    return images, labels


def read_gt_boxes(label_path: Path, image_size: tuple[int, int]) -> list[tuple[float, float, float, float]]:
    width, height = image_size
    if not label_path.exists():
        return []

    boxes: list[tuple[float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, x, y, w, h = [float(value) for value in parts]
        cx, cy, bw, bh = x * width, y * height, w * width, h * height
        boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return boxes


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(
    predictions: list[tuple[float, float, float, float]],
    targets: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> tuple[int, int, int]:
    pairs: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(predictions):
        for target_index, target in enumerate(targets):
            overlap = iou(pred, target)
            if overlap >= iou_threshold:
                pairs.append((overlap, pred_index, target_index))

    matched_predictions: set[int] = set()
    matched_targets: set[int] = set()
    for _, pred_index, target_index in sorted(pairs, reverse=True):
        if pred_index in matched_predictions or target_index in matched_targets:
            continue
        matched_predictions.add(pred_index)
        matched_targets.add(target_index)

    true_positive = len(matched_predictions)
    false_positive = len(predictions) - true_positive
    false_negative = len(targets) - len(matched_targets)
    return true_positive, false_positive, false_negative


def draw_review_image(
    image_path: Path,
    output_path: Path,
    targets: list[tuple[float, float, float, float]],
    predictions: list[tuple[float, float, float, float]],
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in targets:
        draw.rectangle(box, outline=(0, 220, 0), width=4)
    for box in predictions:
        draw.rectangle(box, outline=(255, 40, 40), width=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    thresholds = sorted(float(value.strip()) for value in args.thresholds.split(",") if value.strip())
    min_threshold = min(thresholds)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    images_dir, labels_dir = load_dataset(Path(args.data), args.split)
    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    model = YOLO(args.model)
    stats = {
        threshold: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "positive_images": 0,
            "negative_images": 0,
            "triggered_positive_images": 0,
            "missed_positive_images": 0,
            "false_alarm_negative_images": 0,
        }
        for threshold in thresholds
    }
    export_threshold = min(thresholds, key=lambda value: abs(value - args.export_threshold))
    missed_exports = 0
    false_alarm_exports = 0

    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=min_threshold,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        stream=True,
        verbose=False,
    )

    for image_path, result in zip(image_paths, results):
        with Image.open(image_path) as image:
            targets = read_gt_boxes(labels_dir / f"{image_path.stem}.txt", image.size)

        all_predictions: list[tuple[float, tuple[float, float, float, float]]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            all_predictions = [(float(score), tuple(float(value) for value in box)) for score, box in zip(confs, xyxy)]

        for threshold in thresholds:
            predictions = [box for score, box in all_predictions if score >= threshold]
            tp, fp, fn = match_boxes(predictions, targets, args.iou)
            row = stats[threshold]
            row["tp"] += tp
            row["fp"] += fp
            row["fn"] += fn

            if targets:
                row["positive_images"] += 1
                if tp > 0:
                    row["triggered_positive_images"] += 1
                else:
                    row["missed_positive_images"] += 1
            else:
                row["negative_images"] += 1
                if predictions:
                    row["false_alarm_negative_images"] += 1

            if threshold == export_threshold:
                if targets and tp == 0 and missed_exports < args.max_export:
                    draw_review_image(image_path, output / "missed" / image_path.name, targets, predictions)
                    missed_exports += 1
                if not targets and predictions and false_alarm_exports < args.max_export:
                    draw_review_image(image_path, output / "false_alarms" / image_path.name, targets, predictions)
                    false_alarm_exports += 1

    csv_path = output / "threshold_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "conf",
            "box_precision",
            "box_recall",
            "box_f1",
            "image_recall",
            "negative_false_alarm_rate",
            "tp",
            "fp",
            "fn",
            "positive_images",
            "triggered_positive_images",
            "missed_positive_images",
            "negative_images",
            "false_alarm_negative_images",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for threshold in thresholds:
            row = stats[threshold]
            precision = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0.0
            recall = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            image_recall = (
                row["triggered_positive_images"] / row["positive_images"] if row["positive_images"] else 0.0
            )
            false_alarm_rate = (
                row["false_alarm_negative_images"] / row["negative_images"] if row["negative_images"] else 0.0
            )
            writer.writerow(
                {
                    "conf": f"{threshold:.2f}",
                    "box_precision": f"{precision:.4f}",
                    "box_recall": f"{recall:.4f}",
                    "box_f1": f"{f1:.4f}",
                    "image_recall": f"{image_recall:.4f}",
                    "negative_false_alarm_rate": f"{false_alarm_rate:.4f}",
                    **row,
                }
            )

    if shutil.which("column"):
        print(f"wrote: {csv_path}")
    print("conf  box_P   box_R   box_F1  image_R  neg_FA  missed  false_alarm")
    for threshold in thresholds:
        row = stats[threshold]
        precision = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0.0
        recall = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        image_recall = row["triggered_positive_images"] / row["positive_images"] if row["positive_images"] else 0.0
        false_alarm_rate = row["false_alarm_negative_images"] / row["negative_images"] if row["negative_images"] else 0.0
        print(
            f"{threshold:>4.2f}  {precision:>6.3f}  {recall:>6.3f}  {f1:>6.3f}  "
            f"{image_recall:>7.3f}  {false_alarm_rate:>6.3f}  "
            f"{row['missed_positive_images']:>6}  {row['false_alarm_negative_images']:>11}"
        )
    print(f"review_images: {output}")
    print("green boxes are labels; red boxes are predictions.")


if __name__ == "__main__":
    main()
