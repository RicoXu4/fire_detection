from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.detector import FireDetector
from app.schemas import DetectionResponse
from app.utils import draw_detections, image_to_base64_png, load_image_from_bytes

app = FastAPI(title="Fire Detection AI", version="1.0.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache
def get_detector() -> FireDetector:
    settings = get_settings()
    return FireDetector(
        model_path=settings.model_path,
        conf_threshold=settings.conf_threshold,
        iou_threshold=settings.iou_threshold,
        image_size=settings.image_size,
        device=settings.device,
        allowed_classes=settings.allowed_classes,
        class_conf_thresholds=settings.class_conf_thresholds,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(...),
    visualize: bool = True,
    conf: float | None = Query(default=None, ge=0.01, le=1.0),
    fire_conf: float | None = Query(default=None, ge=0.01, le=1.0),
    smoke_conf: float | None = Query(default=None, ge=0.01, le=1.0),
    label_mode: str = Query(default="risk", pattern="^(raw|risk)$"),
) -> DetectionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    content = await file.read()
    try:
        image = load_image_from_bytes(content)
        detections = get_detector().predict(
            image,
            conf_threshold=conf,
            fire_conf_threshold=fire_conf,
            smoke_conf_threshold=smoke_conf,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    visualization = None
    if visualize:
        visualization = image_to_base64_png(draw_detections(image, detections, label_mode=label_mode))

    height, width = image.shape[:2]
    return DetectionResponse(
        image_width=width,
        image_height=height,
        detections=detections,
        visualization=visualization,
    )
