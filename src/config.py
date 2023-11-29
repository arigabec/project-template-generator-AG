from enum import Enum
from functools import lru_cache
from functools import cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class GPTModel(str, Enum):
    gpt_4 = "gpt-4"
    gpt_3_5_turbo = "gpt-3.5-turbo"

class Settings(BaseSettings):
    service_name: str = "Awesome projects"
    k_revision: str = "Local"
    log_level: str = "DEBUG"
    openai_key: str
    model: GPTModel = GPTModel.gpt_3_5_turbo

    class Config:
        env_file = ".env"

class PredictorSettings(BaseSettings):
    api_name: str = "Object Detection service"
    revision: str = "local"
    yolo_version: str = "yolov8n.pt"
    log_level: str = "DEBUG"
    openai_key: str

    class Config:
        env_file = ".env"

@lru_cache
def get_settings():
    return Settings()

@cache
def get_predictor_settings():
    return PredictorSettings()
