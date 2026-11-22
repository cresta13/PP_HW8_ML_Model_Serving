"""Shared synthetic request examples and test credentials."""

from typing import Final

AUTH: Final[tuple[str, str]] = ("test_user", "test_password")
PAYLOAD: Final[dict[str, int | float]] = {
    "Pregnancies": 2,
    "Glucose": 140,
    "BMI": 35.5,
    "Age": 32,
}
