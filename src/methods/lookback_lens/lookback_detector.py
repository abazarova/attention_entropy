from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.linear_model import LogisticRegression

from src.methods.hallucination_detection_abc import HallucinationDetectionMethod
from src.methods.llm_base import LLMBase

from ..caching_utils import get_dataframe_hash
from .utils import get_lookback_features


@dataclass
class LookbackLens(HallucinationDetectionMethod):
    """Implementation of the Lookback Lens.

    Parameters
    ----------
    model_name : str
        Name of the model to use for embedding extraction
    cache_dir : str
        Directory to cache lookback features
    predefined_span: bool
    sliding_window: int
    n_layers: int
    n_heads: int
    device : str
        Device to run the model on ('cuda' or 'cpu')
    dtype : str
        Data type for model ('float16' or 'float32')

    """

    model_name: str
    cache_dir: str
    sliding_window: int = 8
    n_layers: int = 32
    n_heads: int = 32
    device: str = "cuda"
    dtype: str = "float16"

    _columns: list[str] = field(default=None, init=False)

    def transform(self, X: pd.DataFrame) -> list[torch.Tensor]:
        """Transform attention maps from all layers and heads into an
        LR feature vector of size n_layers x n_heads.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with 'prompt' and 'response' columns

        Returns
        -------
        list[torch.Tensor]
            List of feature vectors, each of shape (n_layers x n_heads, ).

        """
        self._columns = X.columns
        self.hash_name = get_dataframe_hash(X)
        self.llm_model = LLMBase(self.model_name, self.dtype, self.device)
        return X.values.tolist()

    def fit(self, X_train: list[list], y_train: list[int]) -> "LookbackLens":
        """Fit the detector on training data.

        Parameters
        ----------
        X_train : list[torch.Tensor]
            List of samples, where each sample is a tensor of shape [num_layers, hidden_dim]
        y_train : list[int]
            Labels (1 for hallucination, 0 for truthful)

        Returns
        -------
        HaloscopeDetector
            Fitted detector

        """
        logger.info(f"Fitting {self.__class__.__name__} on {len(X_train)} samples")
        X_train = pd.DataFrame(X_train, columns=self._columns)
        spans_available = X_train["name"][0] == "ragtruth"

        # Extract LR features
        data, labels = get_lookback_features(
            X_train,
            y_train,
            llm_model=self.llm_model,
            sliding_window=self.sliding_window,
            spans_available=spans_available,
            cache_dir=Path(self.cache_dir) / self.model_name / self.hash_name
        )
        data = torch.stack(data, dim=0).numpy()
        clf = LogisticRegression()
        self.clf = clf.fit(data, labels)
        return self

    def predict_score(self, X: list[list]) -> np.ndarray:
        """Predict hallucination scores for new samples.

        Parameters
        ----------
        X : list[torch.Tensor]
            List of samples, where each sample is a tensor of shape [num_layers, hidden_dim]

        Returns
        -------
        np.ndarray
            Array of hallucination scores, shape [batch_size]

        """
        X = pd.DataFrame(X, columns=self._columns)

        data, _ = get_lookback_features(
            X,
            llm_model=self.llm_model,
            sliding_window=self.sliding_window,
            spans_available=False,
            cache_dir=Path(self.cache_dir) / self.model_name / self.hash_name
        )  # (n_samples, n_pieces, n_layers * n_heads)
        scores = []
        for elem in data:
            spanwise_scores = self.clf.predict_proba(torch.stack(elem).numpy())[:, 1]
            scores.append(max(spanwise_scores))

        return scores
