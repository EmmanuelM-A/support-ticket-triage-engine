"""
Configuration settings for the TicketTriage API. Each configuration
class handles a specific domain of settings.
"""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)

_DEFAULT_MODEL_CONFIG = SettingsConfigDict(
    env_file=_ENV_FILE,
    env_file_encoding="utf-8",
    env_ignore_empty=True,
    extra="ignore",
)


class _BaseSettings(BaseSettings):
    """
    Shared base settings class for all settings classes in the application.
    """

    @model_validator(mode="before")
    @classmethod
    def _drop_empty_strings(cls, values: dict) -> dict:
        return {k: v for k, v in values.items() if v != ""}


class _ServerSettings(_BaseSettings):
    """
    Server configuration settings for the TicketTriage API application.
    """

    model_config = _DEFAULT_MODEL_CONFIG

    ENV: Literal["development", "production"] = Field(
        default="development",
    )


class _LoggingSettings(_BaseSettings):
    """
    Logging configuration settings for the TicketTriage API application.
    """

    model_config = _DEFAULT_MODEL_CONFIG

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    LOG_FORMAT: str = Field(
        default="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d]: %(message)s"
    )
    DATE_FORMAT: str = Field(default="%Y-%m-%d %H:%M:%S")


class _MachineLearningSettings(_BaseSettings):
    """
    Machine learning configuration settings for the TicketTriage API application.
    """

    model_config = _DEFAULT_MODEL_CONFIG

    RANDOM_SEED: int = Field(default=42)

    SOURCE_DATA_DIR: Path = Field(default=_PROJECT_ROOT / "data" / "source")
    RAW_DATA_DIR: Path = Field(default=_PROJECT_ROOT / "data" / "raw")
    SAMPLES_PER_CATEGORY: int = Field(default=7_000)
    REQUIRED_COLUMNS: list[str] = Field(
        default=[
            "Complaint ID",
            "Date received",
            "Product",
            "Sub-product",
            "Issue",
            "Sub-issue",
            "Consumer complaint narrative",
        ]
    )

    DOWNLOAD_CHUNK_SIZE: int = Field(default=50_000)


class Settings(_BaseSettings):
    """
    Main configuration settings class for the TicketTriage API application.
    """

    model_config = _DEFAULT_MODEL_CONFIG

    server: _ServerSettings = _ServerSettings()
    logging: _LoggingSettings = _LoggingSettings()
    ml: _MachineLearningSettings = _MachineLearningSettings()


settings = Settings()
