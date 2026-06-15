#!/usr/bin/env bash
set -euo pipefail

cd /Users/rico/code/fire_detection

export PYTHONPATH=.
export MPLCONFIGDIR=/Users/rico/code/fire_detection/.cache/matplotlib
export YOLO_CONFIG_DIR=/Users/rico/code/fire_detection/.cache/ultralytics
export FIRE_MODEL_PATH="${FIRE_MODEL_PATH:-runs/detect/runs/train/yolo26n_mixed_indoor_fire_mps_35e/weights/best.pt}"
export FIRE_DEVICE="${FIRE_DEVICE:-cpu}"
export FIRE_IMAGE_SIZE="${FIRE_IMAGE_SIZE:-512}"
export FIRE_FIRE_CONF_THRESHOLD="${FIRE_FIRE_CONF_THRESHOLD:-0.12}"
export FIRE_SMOKE_CONF_THRESHOLD="${FIRE_SMOKE_CONF_THRESHOLD:-0.30}"

mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"

exec .venv310/bin/uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
