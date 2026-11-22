"""FastAPI application, lifecycle and HTTP endpoints."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from app.auth import verify_user
from app.config import Settings
from app.inference import DiabetesModel, InferenceError
from app.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    model_loader: Callable[[Path], DiabetesModel] = DiabetesModel.load,
) -> FastAPI:
    """Build the app with injectable settings and model loader for testing."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Load the model at startup and clear it during shutdown."""
        config = settings if settings is not None else Settings()
        logging.basicConfig(
            level=config.log_level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        logger.setLevel(config.log_level)
        application.state.settings = config
        application.state.model = model_loader(config.model_path)
        logger.info("Model loaded successfully")
        try:
            yield
        finally:
            application.state.model = None
            logger.info("Model session released")

    application = FastAPI(
        title="ML Model Serving API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Return validation errors without echoing user-supplied values."""
        # Do not echo medical data or non-finite values in error responses.
        detail = [
            {key: error[key] for key in ("loc", "msg", "type")}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detail})

    @application.middleware("http")
    async def log_request(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Log endpoint metadata and timing, never sensitive request bodies."""
        started = perf_counter()
        code = 500
        try:
            response = await call_next(request)
            code = response.status_code
            return response
        finally:
            # Route template only: no body, query, headers, login or prediction.
            route = getattr(request.scope.get("route"), "path", "<unmatched>")
            logger.info(
                "method=%s route=%s status=%d duration_ms=%.2f",
                request.method,
                route,
                code,
                (perf_counter() - started) * 1000,
            )

    @application.get("/")
    def root() -> dict[str, str]:
        """Return a public welcome message."""
        return {"message": "Welcome to the ML Model Serving API!"}

    @application.get("/health")
    def health(request: Request) -> dict[str, str]:
        """Report whether the model has been loaded and is ready."""
        if getattr(request.app.state, "model", None) is None:
            raise HTTPException(status_code=503, detail="Model not ready")
        return {"status": "ok"}

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        dependencies=[Depends(verify_user)],
        responses={401: {"description": "Invalid or missing credentials"}},
    )
    def predict(data: PredictionRequest, request: Request) -> PredictionResponse:
        """Return a binary diabetes prediction for an authenticated request."""
        model: DiabetesModel = request.app.state.model
        try:
            result = model.predict(data)
        except InferenceError as exc:
            logger.error("Prediction failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model inference failed",
            ) from exc
        return PredictionResponse(prediction=result)

    return application


app = create_app()
