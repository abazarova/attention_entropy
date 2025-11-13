import typing as tp
from abc import ABC, abstractmethod  # noqa: D100

import numpy as np
import pandas as pd


class HallucinationDetectionDataset(ABC):
    """Abstract class for specifying the interface of the called dataset."""

    @abstractmethod
    def process(
        self,
    ) -> tuple[
        pd.DataFrame, pd.Series, tp.Optional[np.ndarray], tp.Optional[np.ndarray]
    ]:
        """Process and return LLM Answers with hallucination label."""
        pass
