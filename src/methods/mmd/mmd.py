from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from ..caching_utils import cache_result, get_dataframe_hash
from ..customtypes import ModelName
from ..hallucination_detection_abc import HallucinationDetectionMethod
from ..llm_base import LLMBase
from .module import MMDModule


@cache_result(cache_dir="cache/mmd", message="Get MMD")
def get_mmd_scores(
    X: pd.DataFrame,
    model: LLMBase,
    module: MMDModule,
    hash: Optional[str] = None,
    cache_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> list[torch.Tensor]:
    """"""
    mmd_scores = []
    for output, prompt_ids, answer_ids in tqdm(model.generate_llm_outputs(
        X, output_hidden_states=True, output_attentions=False
    ), total=len(X)):
        len_prompt = len(prompt_ids[0])
        hiddens_all_layers = torch.cat(output["hidden_states"][1:], dim=0).float()

        hiddens_prompt = hiddens_all_layers[:, :len_prompt]
        hiddens_answer = hiddens_all_layers[:, len_prompt:]

        scores = -module(hiddens_prompt, hiddens_answer).cpu()
        mmd_scores.append(scores)

    return mmd_scores


@dataclass
class MMD(HallucinationDetectionMethod):
    """MMD class with the desired strategy, model, kernel, and device.

    Parameters
    ----------
    strategy : Literal["supervised", "unsupervised"], optional
        Strategy for training (default is "unsupervised").
    model_name : Literal, optional
        Name of the model to use (default is "Llama-2-7b-chat-hf").
    kernel : Literal, optional
        Kernel type for MMD calculation (default is "rbf").
    kernel_kwargs : dict, optional
        Additional kernel arguments (default is None).
    half : bool, optional
        Whether to load the model in half precision mode (default is True).
    device : str, optional
        Device to run the model on (default is "cuda:0").
    mmd_save_dir : str, optional
        Directory to save the MMD scores (default is "cache/mmd").

    """

    strategy: Literal["supervised", "unsupervised"]
    model_name: ModelName
    kernel: Literal["rbf", "matern"] = "rbf"
    kernel_kwargs: Optional[dict] = None
    dtype: str = "float16"
    device: str = "cuda"
    cache_dir: str = "cache/mmd"

    def __post_init__(self):
        self.clf = None
        self.is_fitted = False

        self.kernel_kwargs = self.kernel_kwargs or {}  # Initialize if None
        self.module = MMDModule(self.kernel, **self.kernel_kwargs).to(self.device)

    def transform(self, X):
        """Calculate the MMD scores in real-time or load precomputed scores.

        Parameters
        ----------
        X : pd.DataFrame
            Input data for MMD calculation.

        Returns
        -------
        torch.Tensor
            MMD scores.

        """
        llm_model = LLMBase(self.model_name, self.dtype, self.device)

        data_hash = get_dataframe_hash(X)
        mmd_cache_name = f"{self.model_name}_{data_hash}_mmd.joblib"
        mmd_scores = get_mmd_scores(
            X,
            llm_model,
            self.module,
            cache_name=mmd_cache_name,
            cache_dir=self.cache_dir,
        )

        return mmd_scores

    def fit(self, X, y, *args) -> "MMD":
        """Fit the MMD model with the given training data and labels.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature data.
        y : pd.Series
            Target labels.

        Returns
        -------
        MMD
            Fitted MMD instance.

        """
        if self.strategy == "unsupervised":
            return self

        self.clf = LogisticRegression().fit(torch.vstack(X).numpy(), y)
        self.is_fitted = True

        return self

    def predict_score(self, X: pd.DataFrame) -> List[float]:
        """Predict MMD scores or probabilities based on the strategy.

        Parameters
        ----------
        X : pd.DataFrame
            Input data for prediction.

        Returns
        -------
        List[float]
            Predictions or MMD scores.

        """
        X = torch.vstack(X).numpy()
        if self.strategy == "supervised" and not self.is_fitted:
            raise Exception("Fit required before prediction.")

        if self.strategy == "supervised":
            pred = self.clf.predict_proba(X)[:, 1]
        else:
            pred = X.mean(axis=1)

        return pred
