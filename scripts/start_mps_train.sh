#!/usr/bin/env bash
set -euo pipefail

cd /Users/rico/code/fire_detection

export PYTHONPATH=.
export MPLCONFIGDIR=/Users/rico/code/fire_detection/.cache/matplotlib
export YOLO_CONFIG_DIR=/Users/rico/code/fire_detection/.cache/ultralytics
export PYTORCH_ENABLE_MPS_FALLBACK=1

mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR" runs/train
exec > >(tee -a /Users/rico/code/fire_detection/runs/train/mps_train.log) 2>&1

echo "=== MPS training start: $(date) ==="

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

model = YOLO("yolo26n.pt")
model.train(
    data="configs/mydata_fire_abs.yaml",
    epochs=20,
    imgsz=416,
    batch=4,
    device="mps",
    project="runs/train",
    name="yolo26n_fire_mps_20e_torch212_ampoff",
    patience=20,
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
