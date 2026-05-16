"""
config.py
---------
Central configuration for the transcription pipeline.
All values can be overridden by environment variables with the same name
(case-insensitive).  Example:
    export WHISPER_MODEL_SIZE=small
    export DEVICE=cuda
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Config(BaseSettings):
    # ------------------------------------------------------------------ #
    # Redis / Celery broker & result-backend                               #
    # ------------------------------------------------------------------ #
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db_broker: int = Field(default=0, alias="REDIS_DB_BROKER")
    redis_db_backend: int = Field(default=1, alias="REDIS_DB_BACKEND")

    @property
    def broker_url(self) -> str:
        return (
            f"redis://{self.redis_host}:{self.redis_port}"
            f"/{self.redis_db_broker}"
        )

    @property
    def result_backend(self) -> str:
        return (
            f"redis://{self.redis_host}:{self.redis_port}"
            f"/{self.redis_db_backend}"
        )

    # ------------------------------------------------------------------ #
    # Whisper inference settings                                           #
    # ------------------------------------------------------------------ #
    whisper_model_size: str = Field(
        default="base", alias="WHISPER_MODEL_SIZE"
    )
    device: str = Field(default="cpu", alias="DEVICE")
    compute_type: str = Field(default="int8", alias="COMPUTE_TYPE")

    # ------------------------------------------------------------------ #
    # Scratch directories                                                  #
    # ------------------------------------------------------------------ #
    uploads_dir: str = Field(
        default="api_scratch/uploads", alias="UPLOADS_DIR"
    )
    outputs_dir: str = Field(
        default="api_scratch/outputs", alias="OUTPUTS_DIR"
    )

    # ------------------------------------------------------------------ #
    # Chunked-upload buffer size (bytes)                                   #
    # ------------------------------------------------------------------ #
    upload_chunk_size: int = Field(
        default=65_536, alias="UPLOAD_CHUNK_SIZE"  # 64 KB
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True


# Module-level singleton – import this everywhere.
settings = Config()
