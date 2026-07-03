from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the single-class fire-risk model on Google Colab.")
    parser.add_argument("--data", default="datasets/mixed_indoor_fire_risk_plus_new/data.yaml")
    parser.add_argument("--source", default="datasets/mixed_indoor_fire_plus_new")
    parser.add_argument("--output-risk-dataset", default="datasets/mixed_indoor_fire_risk_plus_new")
    parser.add_argument("--model", default="", help="Optional explicit starting weight path.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="yolo26n_fire_risk_colab_768_60e")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--drive-output", default="", help="Optional Google Drive directory to copy final run into.")
    return parser.parse_args()


def choose_model(explicit: str) -> str:
    candidates = [
        explicit,
        "runs/detect/runs/train/yolo26n_fire_risk_plus_new_mps_30e/weights/best.pt",
        "runs/detect/runs/train/yolo26n_mixed_indoor_fire_mps_35e/weights/best.pt",
        "yolo26n.pt",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return "yolo26n.pt"


def ensure_dataset(args: argparse.Namespace) -> None:
    data_yaml = Path(args.data)
    if data_yaml.exists():
        print(f"Using existing dataset: {data_yaml}")
        return

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(
            "Missing source dataset. Put datasets/mixed_indoor_fire_plus_new in the project, "
            "or unzip a Drive dataset archive before running this script."
        )

    from scripts.prepare_fire_risk_dataset import main as prepare_main

    import sys

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "prepare_fire_risk_dataset.py",
            "--source",
            str(source),
            "--output",
            args.output_risk_dataset,
        ]
        prepare_main()
    finally:
        sys.argv = old_argv

    if not data_yaml.exists():
        raise SystemExit(f"Expected dataset yaml was not created: {data_yaml}")


def main() -> None:
    args = parse_args()
    ensure_dataset(args)

    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_device:", torch.cuda.get_device_name(0))

    model_path = choose_model(args.model)
    print("starting_model:", model_path)

    model = YOLO(model_path)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        cos_lr=True,
        close_mosaic=15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        amp=True,
        workers=2,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    print(f"results_dir: {save_dir}")

    if args.drive_output:
        destination = Path(args.drive_output) / save_dir.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(save_dir, destination)
        print(f"copied_results_to: {destination}")


if __name__ == "__main__":
    main()
