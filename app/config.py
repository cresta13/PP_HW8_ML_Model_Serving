"""Application settings provided through the environment or a local .env."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODEL_PATH: Path = (
    Path(__file__).resolve().parents[1] / "models" / "diabetes_model.onnx"
)


class Settings(BaseSettings):
    """Validate runtime configuration read from environment or local .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    basic_auth_username: str = Field(min_length=1)
    basic_auth_password: SecretStr = Field(min_length=1)
    model_path: Path = DEFAULT_MODEL_PATH
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("basic_auth_username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        """Reject usernames outside the HTTP Basic compatible character set."""
        if not value.isascii() or ":" in value or not value.isprintable():
            raise ValueError("Username must be printable ASCII without a colon")
        return value

    @field_validator("basic_auth_password")
    @classmethod
    def validate_password(cls, value: SecretStr) -> SecretStr:
        """Reject non-ASCII and control characters in HTTP Basic passwords."""
        password = value.get_secret_value()
        if not password.isascii() or not password.isprintable():
            raise ValueError("Password must be printable ASCII")
        return value
