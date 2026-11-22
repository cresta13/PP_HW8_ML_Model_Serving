"""Shared test fixtures and an in-memory ONNX session double."""

from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.inference import DiabetesModel
from app.main import create_app
from tests.helpers import AUTH


class FakeSession:
    """Emulate the provided classifier's model contract without ONNX Runtime."""

    def __init__(self) -> None:
        """Initialize deterministic probabilities and collect inference calls."""
        self.calls: list[tuple[list[str], dict[str, np.ndarray]]] = []
        self.result: list[object] = [{0: 0.2, 1: 0.8}]
        self.error: Exception | None = None

    def get_inputs(self) -> list[SimpleNamespace]:
        """Return one float32 input containing the four patient features."""
        return [SimpleNamespace(name="input", type="tensor(float)", shape=[None, 4])]

    def get_outputs(self) -> list[SimpleNamespace]:
        """Return the original ONNX label and probability output signatures."""
        return [
            SimpleNamespace(name="output_label", type="tensor(int64)"),
            SimpleNamespace(
                name="output_probability", type="seq(map(int64,tensor(float)))"
            ),
        ]

    def run(
        self,
        output_names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[list[object]]:
        """Record inputs and return the configured single-record probability map."""
        self.calls.append((output_names, inputs))
        if self.error:
            raise self.error
        return [self.result]


@pytest.fixture
def settings() -> Settings:
    """Provide valid credentials without relying on a developer's .env file."""
    return Settings(
        _env_file=None,
        basic_auth_username=AUTH[0],
        basic_auth_password=AUTH[1],
    )


@pytest.fixture
def session() -> FakeSession:
    """Create a new fake model session for each test."""
    return FakeSession()


@pytest.fixture
def client(settings: Settings, session: FakeSession) -> Iterator[TestClient]:
    """Run the API lifespan with the fake session for isolated HTTP tests."""
    application = create_app(settings, model_loader=lambda _: DiabetesModel(session))
    with TestClient(application) as test_client:
        yield test_client
