"""HTTP Basic authentication behavior and OpenAPI security contract."""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakeSession
from tests.helpers import AUTH, PAYLOAD


@pytest.mark.parametrize(
    "auth", [None, ("bad_user", AUTH[1]), (AUTH[0], "bad_password"), ("", "")]
)
def test_unauthorized(
    client: TestClient,
    session: FakeSession,
    auth: tuple[str, str] | None,
) -> None:
    """Reject missing and incorrect credentials without invoking the model."""
    response = client.post("/predict", json=PAYLOAD, auth=auth)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"].startswith("Basic")
    assert not session.calls


@pytest.mark.parametrize("header", ["Bearer token", "Basic !!!", "Basic bm9jb2xvbg=="])
def test_malformed_auth(client: TestClient, session: FakeSession, header: str) -> None:
    """Reject malformed authentication headers before processing the request."""
    response = client.post(
        "/predict",
        json=PAYLOAD,
        headers={"Authorization": header},
    )
    assert response.status_code == 401
    assert not session.calls


def test_openapi_declares_basic_auth(client: TestClient) -> None:
    """The API contract marks prediction as protected and root as public."""
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/predict"]["post"]
    assert {"HTTPBasic": []} in operation["security"]
    assert schema["components"]["securitySchemes"]["HTTPBasic"]["scheme"] == "basic"
    assert "security" not in schema["paths"]["/"]["get"]
