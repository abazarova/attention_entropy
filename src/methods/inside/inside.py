import os
from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import trange

from ..caching_utils import get_dataframe_hash
from ..extract_states import get_generated_responses, get_hidden_states
from ..hallucination_detection_abc import HallucinationDetectionMethod
from ..llm_base import LLMBase


@dataclass
class INSIDE(HallucinationDetectionMethod):
    """
    INSIDE class for extracting hidden states from a language model, caching them, and using them
    to predict hallucinations based on eigenscores of covariance matrices.
    """

    layer: int = 16
    regularization_alpha: float = 1e-3
    aggregation: Literal["max", "min", "mean", "last"] = "last"
    model_name: Literal["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.1"] = (
        "Llama-2-7b-chat-hf"
    )
    dtype: Literal["float32", "float16", "bfloat16"] = "float16"
    device: str = "cuda"
    cache_dir: str = "cache/hiddens"
    generated_responses_dir: str = "cache/generated_responses"

    num_return_sequences: int = 15
    temperature: float = 1.0
    max_new_tokens: int = 512

    def aggregate(self, hiddens: list[torch.Tensor]) -> list[torch.Tensor]:
        """Aggregate hiddens from encoded prompt+answer."""
        if self.aggregation == "mean":
            return hiddens.mean(dim=0)
        if self.aggregation == "min":
            return hiddens.min(dim=0).values
        if self.aggregation == "max":
            return hiddens.max(dim=0).values
        if self.aggregation == "last":
            return hiddens[-1]

        raise ValueError(f"Unsupported aggregation method: {self.aggregation}")

    def transform(self, X: pd.DataFrame) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Generate responses and get hidden states."""

        llm = LLMBase(
            self.model_name,
            self.dtype,
            self.device,
            num_return_sequences=self.num_return_sequences,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )

        data_hash = get_dataframe_hash(X)

        cachefile_general_name = f"{self.model_name}_{data_hash}"

        generated_responses_cache_name = (
            cachefile_general_name + "_generated_responses.joblib"
        )

        generated_responses = get_generated_responses(
            X,
            llm,
            cache_name=generated_responses_cache_name,
            cache_dir=self.generated_responses_dir,
        )

        prompt_list = []
        response_list = []
        for prompt, responses in zip(X["prompt"], generated_responses):
            prompt_list.extend(len(responses) * [prompt])
            response_list.extend(responses)

        new_df = pd.DataFrame({"prompt": prompt_list, "response": response_list})

        data_hash = get_dataframe_hash(new_df)

        cachefile_general_name = f"{self.model_name}_{data_hash}_{self.layer}"

        hiddens_cache_name = cachefile_general_name + "_hiddens.joblib"

        hiddens = get_hidden_states(
            new_df,
            self.layer,
            llm,
            cache_name=hiddens_cache_name,
            cache_dir=self.cache_dir,
            move_to_cpu=False
        )

        del llm

        hiddens_agg = torch.cat([self.aggregate(h) for h in hiddens])
        hiddens_reshaped = hiddens_agg.view(len(X), self.num_return_sequences, -1)

        return hiddens_reshaped

    def fit(self, X: pd.DataFrame, y: pd.Series, *args) -> "INSIDE":
        """
        Placeholder method for fitting the model.

        Parameters:
        ----------
        X : pd.DataFrame
            Feature data for fitting.
        y : pd.Series
            Target labels for fitting.

        Returns:
        -------
        self : INSIDE
            The fitted INSIDE instance.
        """
        return self

    def predict_score(self, X: List[float]) -> np.ndarray[float]:
        hiddens_all = torch.stack(X, dim=0).float()
        
        _, singular_values, _ = torch.linalg.svd(hiddens_all, full_matrices=False)
        singular_values = torch.clamp(singular_values, min=self.regularization_alpha)
        eigenscore = torch.log(singular_values ** 2).mean(dim=1)
        
        return eigenscore.numpy()

    def get_cov(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the covariance matrices of the hidden states for each prompt.

        Parameters:
        ----------
        X : torch.Tensor
            Tensor containing hidden states.

        Returns:
        -------
        torch.Tensor
            Covariance matrices for each prompt.
        """
        mean = torch.mean(X, dim=-1).float()
        centered_sen = X - mean[..., None]
        d = X.shape[-1]
        covariance_matrices = torch.einsum(
            "qin,qjn->qij", centered_sen, centered_sen
        ) / (d - 1)

        return covariance_matrices
