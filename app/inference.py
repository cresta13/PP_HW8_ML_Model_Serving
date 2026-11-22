"""ONNX model adapter: one session per application process."""

from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from app.schemas import PredictionRequest

if TYPE_CHECKING:
    from onnxruntime import InferenceSession


class InferenceError(RuntimeError):
    """Inference failed or the model violated its output contract."""


class DiabetesModel:
    """Adapt the supplied ONNX classifier to the API prediction contract."""

    PROBABILITY_OUTPUT: ClassVar[str] = "output_probability"
    THRESHOLD: ClassVar[float] = 0.5

    def __init__(self, session: "InferenceSession") -> None:
        """Inspect the model signature once and reject unsupported models."""
        self._session = session
        inputs = session.get_inputs()
        if (
            len(inputs) != 1
            or inputs[0].type != "tensor(float)"
            or len(inputs[0].shape) != 2
            or inputs[0].shape[1] != 4
            or (isinstance(inputs[0].shape[0], int) and inputs[0].shape[0] != 1)
        ):
            raise ValueError("Expected one float32 input with shape [batch, 4]")
        outputs = {output.name: output.type for output in session.get_outputs()}
        if outputs.get(self.PROBABILITY_OUTPUT) != "seq(map(int64,tensor(float)))":
            raise ValueError("Expected the original model's probability map output")
        self._input_name = inputs[0].name

    @classmethod
    def load(cls, path: Path) -> "DiabetesModel":
        """Create a single CPU ONNX Runtime session for the supplied model."""
        # Local import keeps test doubles usable without the native runtime.
        import onnxruntime as ort

        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return cls(session)

    def predict(self, data: PredictionRequest) -> Literal[0, 1]:
        """Classify using class-1 probability with a strictly greater-than threshold."""
        # Build a new tensor for each request, in the model's feature order.
        tensor = np.array(
            [[data.Pregnancies, data.Glucose, data.BMI, data.Age]],
            dtype=np.float32,
        )
        try:
            (batch,) = self._session.run(
                [self.PROBABILITY_OUTPUT],
                {self._input_name: tensor},
            )
            if len(batch) != 1 or not isinstance(batch[0], Mapping):
                raise ValueError("Expected one probability map")
            probability = float(batch[0][1])
            if not isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("Invalid probability")
        except Exception as exc:
            raise InferenceError("Model inference failed") from exc
        return 1 if probability > self.THRESHOLD else 0
