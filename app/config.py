from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_path: str = Field(
        "runs/detect/runs/train/yolo26n_mixed_indoor_fire_mps_35e/weights/best.pt",
        alias="FIRE_MODEL_PATH",
    )
    conf_threshold: float = Field(0.20, alias="FIRE_CONF_THRESHOLD")
    fire_conf_threshold: float = Field(0.12, alias="FIRE_FIRE_CONF_THRESHOLD")
    smoke_conf_threshold: float = Field(0.30, alias="FIRE_SMOKE_CONF_THRESHOLD")
    iou_threshold: float = Field(0.45, alias="FIRE_IOU_THRESHOLD")
    image_size: int = Field(512, alias="FIRE_IMAGE_SIZE")
    device: str = Field("cpu", alias="FIRE_DEVICE")
    classes: str = Field("fire,flame,smoke", alias="FIRE_CLASSES")

    model_config = SettingsConfigDict(populate_by_name=True)

    @property
    def allowed_classes(self) -> list[str]:
        if not self.classes:
            return []
        return [item.strip().lower() for item in self.classes.split(",") if item.strip()]

    @property
    def class_conf_thresholds(self) -> dict[str, float]:
        return {
            "fire": self.fire_conf_threshold,
            "flame": self.fire_conf_threshold,
            "smoke": self.smoke_conf_threshold,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
