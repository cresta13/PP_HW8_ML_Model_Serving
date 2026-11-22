"""Request and response contracts for the prediction endpoint."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# The original model takes a float32 tensor; reject values beyond that range.
FLOAT32_MAX: float = 3.4028234663852886e38
NonNegativeFloat = Annotated[float, Field(ge=0, le=FLOAT32_MAX, allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(ge=0, le=int(FLOAT32_MAX))]


class PredictionRequest(BaseModel):
    """Strictly validated patient features in the model-required order."""

    model_config = ConfigDict(strict=True, extra="forbid")

    Pregnancies: NonNegativeInt
    Glucose: NonNegativeFloat
    BMI: NonNegativeFloat
    Age: NonNegativeInt


class PredictionResponse(BaseModel):
    """Binary classification result returned by POST /predict."""

    prediction: Literal[0, 1]
