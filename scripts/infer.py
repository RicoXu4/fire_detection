from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.detector import FireDetector
from app.utils import draw_detections, load_image, save_visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fire detection on one image.")
    parser.add_argument("--model", default="weights/yolov26_fire.pt", help="YOLO v26 compatible weight path.")
    parser.add_argument("--source", required=True, help="Input image path.")
    parser.add_argument("--output", default="runs/predict", help="Output directory.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or GPU index.")
    parser.add_argument("--classes", default="fire,flame,smoke", help="Comma separated allowed class names. Empty keeps all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = [item.strip().lower() for item in args.classes.split(",") if item.strip()]
    image = load_image(args.source)
    detector = FireDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        image_size=args.imgsz,
        device=args.device,
        allowed_classes=classes,
    )
    detections = detector.predict(image)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.source).stem

    json_path = output_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps([det.model_dump() for det in detections], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    visualized = draw_detections(image, detections)
    save_visualization(output_dir / f"{stem}_vis.jpg", visualized)
    print(json.dumps({"detections": len(detections), "json": str(json_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

