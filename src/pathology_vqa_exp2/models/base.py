from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..records import ExperimentCase


class ModelAdapter(ABC):
    model_key: str
    model_id: str

    @abstractmethod
    def generate(self, case: ExperimentCase) -> str:
        """Generate only the model's continuation for one experiment case."""

    def metadata(self) -> dict[str, Any]:
        return {"model_key": self.model_key, "model_id": self.model_id}

