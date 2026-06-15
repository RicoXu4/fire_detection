from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from app.schemas import BoundingBox, Detection


class FireDetector:
    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: str = "cpu",
        allowed_classes: list[str] | None = None,
        class_conf_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.allowed_classes = {name.lower() for name in (allowed_classes or [])}
        self.class_conf_thresholds = {key.lower(): value for key, value in (class_conf_thresholds or {}).items()}
        self.model = YOLO(self.model_path)

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float | None = None,
        fire_conf_threshold: float | None = None,
        smoke_conf_threshold: float | None = None,
    ) -> list[Detection]:
        active_thresholds = dict(self.class_conf_thresholds)
        if fire_conf_threshold is not None:
            active_thresholds["fire"] = fire_conf_threshold
            active_thresholds["flame"] = fire_conf_threshold
        if smoke_conf_threshold is not None:
            active_thresholds["smoke"] = smoke_conf_threshold
        base_conf = conf_threshold
        if base_conf is None and active_thresholds:
            base_conf = min(active_thresholds.values())
        if base_conf is None:
            base_conf = self.conf_threshold

        results = self.model.predict(
            source=image,
            conf=base_conf,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names or self.model.names
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls.item())
            label = str(names.get(class_id, class_id)).lower()
            if self.allowed_classes and label not in self.allowed_classes:
                continue
            confidence = float(box.conf.item())
            if confidence < active_thresholds.get(label, base_conf):
                continue
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    label=label,
                    class_id=class_id,
                    confidence=confidence,
                    box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        return detections
