"""HTTP Basic authentication for protected endpoints."""

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()


def verify_user(
    request: Request,
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
) -> None:
    """Require valid HTTP Basic credentials for a protected API endpoint."""
    settings = request.app.state.settings
    username_ok = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        settings.basic_auth_username.encode("utf-8"),
    )
    password_ok = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        settings.basic_auth_password.get_secret_value().encode("utf-8"),
    )
    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
