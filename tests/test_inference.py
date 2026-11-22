"""Inference threshold, model contract and real ONNX integration tests."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.inference import DiabetesModel, InferenceError
from app.main import create_app
from app.schemas import PredictionRequest
from tests.conftest import FakeSession
from tests.helpers import AUTH, PAYLOAD


@pytest.mark.parametrize(
    "probability,expected", [(0, 0), (0.499, 0), (0.5, 0), (0.501, 1), (1, 1)]
)
def test_threshold(session: FakeSession, probability: float, expected: int) -> None:
    """Classify as 1 only when class-1 probability is strictly over 0.5."""
    session.result = [{0: 1 - probability, 1: probability}]
    assert DiabetesModel(session).predict(PredictionRequest(**PAYLOAD)) == expected


@pytest.mark.parametrize(
    "result", [[], [{0: 1}], [{1: float("nan")}], [{1: -1}], [{1: 2}], [0]]
)
def test_invalid_model_output(session: FakeSession, result: list[object]) -> None:
    """Reject invalid model probability payloads."""
    session.result = result
    with pytest.raises(InferenceError):
        DiabetesModel(session).predict(PredictionRequest(**PAYLOAD))


def test_invalid_model_input_contract(session: FakeSession) -> None:
    """Reject model signatures with an unexpected number of features."""
    session.get_inputs = lambda: [
        SimpleNamespace(name="input", type="tensor(float)", shape=[None, 5])
    ]
    with pytest.raises(ValueError, match="float32 input"):
        DiabetesModel(session)


def test_invalid_model_output_contract(session: FakeSession) -> None:
    """Reject model signatures without the required probability output."""
    session.get_outputs = lambda: []
    with pytest.raises(ValueError, match="probability map"):
        DiabetesModel(session)


@pytest.mark.integration
@pytest.mark.parametrize(
    "payload,expected",
    [(PAYLOAD, 1), (dict.fromkeys(PAYLOAD, 0), 0)],
)
def test_original_model(
    settings: Settings,
    payload: dict[str, int | float],
    expected: int,
) -> None:
    """Evaluate the unchanged professor-supplied ONNX model directly."""
    model = DiabetesModel.load(settings.model_path)
    assert model.predict(PredictionRequest(**payload)) == expected


@pytest.mark.integration
def test_real_api(settings: Settings) -> None:
    """Make authenticated requests through the real FastAPI and ONNX stack."""
    with TestClient(create_app(settings)) as client:
        response = client.post("/predict", json=PAYLOAD, auth=AUTH)
        assert response.status_code == 200
        assert response.json() == {"prediction": 1}
        assert client.post("/predict", json=PAYLOAD).status_code == 401
