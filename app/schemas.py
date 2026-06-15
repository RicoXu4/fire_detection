from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    label: str
    class_id: int
    confidence: float
    box: BoundingBox


class DetectionResponse(BaseModel):
    image_width: int
    image_height: int
    detections: list[Detection] = Field(default_factory=list)
    visualization: str | None = None

