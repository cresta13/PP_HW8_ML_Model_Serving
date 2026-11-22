"""HTTP endpoint tests: validation, logging, lifecycle and concurrency."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from httpx import Response

from app.config import Settings
from app.inference import DiabetesModel
from app.main import create_app
from tests.conftest import FakeSession
from tests.helpers import AUTH, PAYLOAD


def test_root(client: TestClient) -> None:
    """The public root returns a welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health(client: TestClient) -> None:
    """A successfully loaded model is reported as ready."""
    assert client.get("/health").json() == {"status": "ok"}


def test_predict(client: TestClient, session: FakeSession) -> None:
    """Authorized requests pass the correct float32 tensor to the model."""
    response = client.post("/predict", json=PAYLOAD, auth=AUTH)
    assert response.status_code == 200
    assert response.json() == {"prediction": 1}
    names, inputs = session.calls[0]
    assert names == ["output_probability"]
    assert inputs["input"].dtype == np.float32
    np.testing.assert_array_equal(inputs["input"], [[2, 140, 35.5, 32]])


@pytest.mark.parametrize(
    "changes",
    [
        {"Age": -1},
        {"Pregnancies": -1},
        {"BMI": -1},
        {"Glucose": -1},
        {"Age": 2.5},
        {"Pregnancies": True},
        {"Age": "32"},
        {"BMI": "35.5"},
        {"BMI": None},
        {"extra": 1},
        {"BMI": 1e39},
    ],
)
def test_invalid_data(
    client: TestClient,
    session: FakeSession,
    changes: dict[str, object],
) -> None:
    """Invalid values are rejected before any inference is attempted."""
    response = client.post("/predict", json=PAYLOAD | changes, auth=AUTH)
    assert response.status_code == 422
    assert not session.calls


def test_missing_field(client: TestClient, session: FakeSession) -> None:
    """Required model features cannot be omitted."""
    data = {key: value for key, value in PAYLOAD.items() if key != "Age"}
    assert client.post("/predict", json=data, auth=AUTH).status_code == 422
    assert not session.calls


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_data(client: TestClient, session: FakeSession, value: str) -> None:
    """Nonfinite numeric values never reach the model."""
    response = client.post(
        "/predict",
        content=f'{{"Pregnancies":2,"Glucose":140,"BMI":{value},"Age":32}}',
        headers={"Content-Type": "application/json"},
        auth=AUTH,
    )
    assert response.status_code == 422
    assert not session.calls


def test_malformed_json(client: TestClient, session: FakeSession) -> None:
    """Malformed JSON produces an HTTP validation error."""
    response = client.post(
        "/predict",
        content="{",
        headers={"Content-Type": "application/json"},
        auth=AUTH,
    )
    assert response.status_code == 422
    assert not session.calls


def test_model_failure_is_sanitized(client: TestClient, session: FakeSession) -> None:
    """Runtime details from the model are not exposed to API callers."""
    session.error = RuntimeError("private internal details")
    response = client.post("/predict", json=PAYLOAD, auth=AUTH)
    assert response.status_code == 500
    assert response.json() == {"detail": "Model inference failed"}


def test_logging_without_sensitive_data(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only HTTP metadata is logged, not sensitive patient or auth data."""
    with caplog.at_level("INFO", logger="app.main"):
        client.post(
            "/predict?secret=hidden",
            json=PAYLOAD | {"BMI": 35.5123456},
            auth=AUTH,
        )
    assert "method=POST route=/predict status=200" in caplog.text
    assert "test_password" not in caplog.text
    assert "test_user" not in caplog.text
    assert "hidden" not in caplog.text
    assert "35.5123456" not in caplog.text
    assert "prediction" not in caplog.messages[0]


def test_load_once_and_release(settings: Settings, session: FakeSession) -> None:
    """The model session is initialized once and released at shutdown."""
    loader = Mock(return_value=DiabetesModel(session))
    application = create_app(settings, model_loader=loader)
    with TestClient(application) as client:
        for _ in range(3):
            assert client.post("/predict", json=PAYLOAD, auth=AUTH).status_code == 200
        loader.assert_called_once_with(settings.model_path)
    assert application.state.model is None


def test_concurrent_requests_keep_inputs_isolated(
    client: TestClient,
    session: FakeSession,
) -> None:
    """Concurrent requests use separate tensors without a data race."""
    barrier = Barrier(2, timeout=5)

    def synchronized_run(
        names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[list[dict[int, float]]]:
        """Synchronize infer calls to check that input values remain isolated."""
        barrier.wait()
        glucose = inputs["input"][0, 1]
        probability = 0.9 if glucose > 100 else 0.1
        return [[{0: 1 - probability, 1: probability}]]

    session.run = synchronized_run

    def predict(glucose: int) -> Response:
        """Send a prediction request for the specified glucose value."""
        return client.post("/predict", json=PAYLOAD | {"Glucose": glucose}, auth=AUTH)

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(predict, [50, 150]))
    assert [response.status_code for response in responses] == [200, 200]
    assert [response.json() for response in responses] == [
        {"prediction": 0},
        {"prediction": 1},
    ]
