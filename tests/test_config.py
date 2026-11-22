"""Environment settings and validation tests."""

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_env_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read credentials from environment without revealing passwords in repr."""
    monkeypatch.setenv("BASIC_AUTH_USERNAME", "env_user")
    monkeypatch.setenv("BASIC_AUTH_PASSWORD", "env_password")
    settings = Settings(_env_file=None)
    assert settings.basic_auth_username == "env_user"
    assert settings.basic_auth_password.get_secret_value() == "env_password"
    assert "env_password" not in repr(settings)


def test_required_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail configuration when the required auth credentials are missing."""
    monkeypatch.delenv("BASIC_AUTH_USERNAME", raising=False)
    monkeypatch.delenv("BASIC_AUTH_PASSWORD", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("username", ["user:name", "логин", "", "user\nname"])
def test_invalid_username(username: str) -> None:
    """Reject invalid or unrepresentable HTTP Basic usernames."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, basic_auth_username=username, basic_auth_password="pw")
