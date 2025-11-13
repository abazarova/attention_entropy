import json
import os
import typing as tp
from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import pandas as pd
import torch

from ..caching_utils import get_dataframe_hash
from ..customtypes import ModelName
from ..extract_states import get_token_distributions
from ..hallucination_detection_abc import HallucinationDetectionMethod
from ..llm_base import LLMBase

T = list[float]


def compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy from logits.

    Parameters:
    ----------
    logits : torch.Tensor
        Logits from the model.

    Returns:
    -------
    torch.Tensor
        Entropy values.
    """
    probabilities = torch.softmax(logits, dim=-1)
    log_probabilities = torch.log(probabilities + 1e-6)
    entropies = -torch.sum(probabilities * log_probabilities, dim=-1)
    return entropies


@dataclass
class TokenwiseEntropy(HallucinationDetectionMethod):
    """
    A class to compute token-wise entropy using pre-trained models.

    This class provides methods to calculate token-wise entropy for given text prompts and responses using
    pre-trained language models. It supports different aggregation methods for combining entropy values and
    allows saving and loading of precomputed entropy results.

    Attributes:
    ----------
    aggregation : Literal['max', 'min', 'mean']
        The method to aggregate the entropy scores. Options are 'max', 'min', or 'mean'.
    model_name : Literal["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.1"]
        The name of the pre-trained model to use for computing entropies. Choices are 'Llama-2-7b-chat-hf' or
        'Mistral-7B-Instruct-v0.1'.
    half : bool
        Whether to use half precision for the model. Default is True. Using half precision can reduce memory
        usage but may affect model performance.
    device : str
        The device to run the model on, e.g., 'cuda:0' for GPU or 'cpu' for CPU. Default is 'cuda:0'.
    entropy_save_dir : str
        Directory to save or load precomputed entropy data. If this directory does not exist, it will be created.
        Default is 'cache/tokenwise_entropy'.
    dataset_name : str
        The name of the dataset being processed, used for naming the entropy files. Default is 'RAGTruth'.
    """

    aggregation: tp.Literal["max", "min", "mean"]
    model_name: ModelName
    dtype: Literal["float32", "float16", "bfloat16"] = ("float16",)
    device: str = ("cuda",)
    cache_dir: str = ("cache/token_distribution",)

    def transform(self, X: pd.DataFrame) -> tp.Sequence[T]:
        """
        Calculate token-wise entropy for each entry in the DataFrame.

        Parameters:
        ----------
        X : pd.DataFrame
            DataFrame containing 'prompt' and 'response' columns.

        Returns:
        -------
        list
            List of entropy values for each entry.
        """

        llm_model = LLMBase(self.model_name, self.dtype, self.device)

        data_hash = get_dataframe_hash(X)
        distributions_cache_name = (
            f"{self.model_name}_{data_hash}" + "_token_distributions.joblib"
        )
        token_distributions = get_token_distributions(
            X, llm_model, cache_name=distributions_cache_name, cache_dir=self.cache_dir
        )

        entropies_list = []

        for token_distribution in token_distributions:
            entropies_list.append(compute_entropy_from_logits(token_distribution))

        return entropies_list

    def fit(self, X_train: list[T], y_train: pd.Series, *args):
        return self

    def predict_score(self, X: list[T]) -> np.ndarray[float]:
        """
        Perform inference, load or compute entropies, and aggregate results.

        Parameters:
        ----------
        X : pd.DataFrame
            DataFrame containing 'prompt' and 'response' columns.

        Returns:
        -------
        list
            Aggregated entropy values.
        """

        # Aggregate the entropies
        if self.aggregation == "max":
            aggregated_entropy = [max(sample) for sample in X]
        elif self.aggregation == "min":
            aggregated_entropy = [min(sample) for sample in X]
        elif self.aggregation == "mean":
            aggregated_entropy = [sum(sample) / len(sample) for sample in X]
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation}")

        return aggregated_entropy
