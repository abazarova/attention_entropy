import gc
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score
from tqdm import trange

from ..extract_states import get_attention_maps
from ..hallucination_detection_abc import HallucinationDetectionMethod
from ..llm_base import LLMBase

logger.disable("src.methods.caching_utils")


@dataclass
class LLMCheck(HallucinationDetectionMethod):
    """An implementation of LLM-Check (attention scores): https://neurips.cc/virtual/2024/poster/95584.

    Attributes
    ----------
    model_name : Literal["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.1"]
        The name of the pre-trained model to use. Choices are 'Llama-2-7b-chat-hf' or
        'Mistral-7B-Instruct-v0.1'.
    dtype : str
        The data type used for the LLM inference, e.g., 'float16', 'float32'. Determines the precision of computations.
    device : str
        The device to run the model on, such as 'cuda' for GPU or 'cpu' for CPU. Default is 'cuda'.
    cache_dir : str
        Directory to save or load precomputed MTopDiv data. If this directory does not exist, it will be created.
        Default is 'cache/mtopdiv'.
    n_layers : int
        Number of layers in the considered pre-trained model.

    """

    model_name: Literal["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.1"]
    dtype: str = "float16"
    device: str = "cuda"
    cache_dir: str = "cache/llm_check"

    n_layers: int = 32

    # Interanl state
    _best_layer: int = field(default=None, init=False)

    def __post_init__(self):
        """Post initializtion."""
        self.llm_model = LLMBase(self.model_name, self.dtype, self.device)

    def transform(self, X: pd.DataFrame) -> list[list[float]]:
        """Calculate attention score for each entry in the DataFrame.

        Calculation is performed for the layer selected on the validation set.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing 'prompt' and 'response' columns.

        Returns
        -------
        list
            Attention score (average logdet of attention maps) for each entry.

        """
        if self._best_layer is None:
            raise TypeError(
                "Best layer is not set. Run select_layer on the validation set first."
            )
        attention_maps = get_attention_maps(X, layer=self._best_layer, model=self.llm_model)
        attention_scores = self.logdet(attention_maps)
        return attention_scores

    def fit(
        self,
        X_train: list[list[float]],
        y_train: list[int],
        *args
    ) -> "LLMCheck":
        """Do nothing, is needed just to fit the method into the pipeline."""
        return self

    def predict_score(self, X: list[float]) -> np.ndarray[float]:
        """Perform inference.

        Parameters
        ----------
        X : list[float]
            List of attention scores of length n_samples.

        Returns
        -------
        List[float]
            List of attention scores of length n_samples.

        """
        return X

    def _fix_zeros(self, attention_map: torch.Tensor) -> torch.Tensor:
        attention_map[attention_map < 1e-5] = 1e-5
        return attention_map

    def logdet(self, attention_maps: list[torch.Tensor]) -> list[float]:
        """Calculate attention scores for a list of attention maps.

        Args:
            attention_maps (list[torch.Tensor]): list of length (n_samples) containing tensors
            of shape (n_heads, n_tokens, n_tokens).

        Returns:
            list[float]: list of attention scores.

        """
        return [
            torch.log(
                torch.diagonal(self._fix_zeros(attn_map.float()), dim1=-2, dim2=-1)
            )
            .mean()
            .item()
            for attn_map in attention_maps
        ]

    def fit_hyperparameters(self, X_val: pd.DataFrame, y_val: list[int]):
        """Select best layer on the validation set."""
        best_roc_auc = 0

        for layer in trange(self.n_layers):
            attention_maps = get_attention_maps(
                X_val, layer=layer, model=self.llm_model
            )  # (n_samples, n_heads, m, m)
            attention_scores = self.logdet(attention_maps)
            roc_auc = roc_auc_score(y_val, attention_scores)
            print(f"Layer: {layer}, AUROC: {roc_auc:.3f}")

            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                self._best_layer = layer
            del attention_maps
            torch.cuda.empty_cache()  # Clear GPU cache
            gc.collect()

        print(f"Best ROC-AUC score,{best_roc_auc:.3f}\nBest layer: {self._best_layer}")
