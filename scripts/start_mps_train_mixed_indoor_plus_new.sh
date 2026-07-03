#!/usr/bin/env bash
set -euo pipefail

cd /Users/rico/code/fire_detection

export PYTHONPATH=.
export MPLCONFIGDIR=/Users/rico/code/fire_detection/.cache/matplotlib
export YOLO_CONFIG_DIR=/Users/rico/code/fire_detection/.cache/ultralytics
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1

mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR" runs/train

LOG=/Users/rico/code/fire_detection/runs/train/mps_mixed_indoor_plus_new_fire.log
exec >> "$LOG" 2>&1

main() {
  echo "=== Mixed indoor + new fire MPS training start: $(date) ==="

  if [[ ! -f datasets/mixed_indoor_fire_plus_new/data.yaml ]]; then
    .venv310/bin/python scripts/prepare_mixed_indoor_plus_new_dataset.py
  else
    echo "dataset already exists: datasets/mixed_indoor_fire_plus_new/data.yaml"
  fi

  .venv310/bin/python - <<'PY'
import platform
import torch

print("python_macos:", platform.mac_ver()[0])
print("torch:", torch.__version__)
print("mps_built:", torch.backends.mps.is_built())
print("mps_available:", torch.backends.mps.is_available())

if not torch.backends.mps.is_available():
    raise SystemExit("MPS is not available in this process; refusing to train on CPU.")
PY

  .venv310/bin/python - <<'PY'
from ultralytics import YOLO

model = YOLO("runs/detect/runs/train/yolo26n_mixed_indoor_fire_mps_35e/weights/best.pt")
model.train(
    data="datasets/mixed_indoor_fire_plus_new/data.yaml",
    epochs=30,
    imgsz=512,
    batch=4,
    device="mps",
    project="runs/train",
    name="yolo26n_mixed_indoor_plus_new_mps_30e",
    patience=10,
    cos_lr=True,
    close_mosaic=10,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    amp=False,
    workers=0,
)
PY
}

main
