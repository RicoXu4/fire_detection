from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.schemas import Detection


def load_image_from_bytes(content: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(content)).convert("RGB")
    return np.array(image)


def load_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image)


def draw_detections(image: np.ndarray, detections: list[Detection], label_mode: str = "raw") -> np.ndarray:
    canvas = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(round, (det.box.x1, det.box.y1, det.box.x2, det.box.y2))
        is_risk = label_mode == "risk" and det.label in {"fire", "flame", "smoke"}
        display_label = "fire risk" if is_risk else det.label
        color = (235, 72, 47) if display_label != "smoke" else (80, 120, 150)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{display_label} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(y1 - 8, th + 8)
        cv2.rectangle(canvas, (x1, y_text - th - 8), (x1 + tw + 8, y_text + 4), color, -1)
        cv2.putText(canvas, text, (x1 + 4, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return canvas


def save_visualization(path: str | Path, image: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output)


def image_to_base64_png(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("failed to encode visualization")
    return base64.b64encode(encoded.tobytes()).decode("ascii")
