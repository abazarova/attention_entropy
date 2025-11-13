from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
from tqdm import trange

from ..caching_utils import cache_result, get_dataframe_hash
from ..extract_states import get_generated_responses
from ..hallucination_detection_abc import HallucinationDetectionMethod
from ..llm_base import LLMBase
from ..performance_monitor import log_execution_time

nltk.download("punkt")


@log_execution_time(use_cuda_timing=True)
@cache_result(cache_dir="cache/sent_scores", message="Get sentence scores")
def get_sent_scores(
    X: pd.DataFrame,
    model: SelfCheckNLI,
    model_name: str,
    hash: Optional[str] = None,
    cache_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> list[torch.Tensor]:
    """"""

    # remove special tokens
    def spec_token(model_name):
        if model_name in [
            "Mistral-7B-Instruct-v0.1",
            "Llama-2-7b-chat-hf",
            "Llama-2-13b-chat-hf",
        ]:
            return "</s>"
        elif model_name in ["Llama-3.1-8B-Instruct"]:
            return "<|eot_id|>"
        else:
            return "<|endoftext|>"

    eos_token = spec_token(model_name)
    X["response"] = X["response"].apply(lambda x: x.replace(eos_token, ""))
    X["generated_responses"] = X["generated_responses"].apply(
        lambda x: [r.replace(eos_token, "") for r in x]
    )
    scores = []
    for i in trange(len(X)):
        sentences = sent_tokenize(X["response"].iloc[i])

        sent_scores_selfcheck_nli = model.predict(
            sentences=sentences,  # list of sentences
            sampled_passages=X["generated_responses"].iloc[
                i
            ],  # list of sampled passages
        )

        scores.append(list(sent_scores_selfcheck_nli))

    return scores


@dataclass
class CustomSelfCheckNLI(HallucinationDetectionMethod):
    """
    Custom class to perform sentence-level Natural Language Inference (NLI) using the SelfCheckNLI model.
    """

    aggregation: Literal["min", "max", "mean"] = "mean"
    model_name: Literal["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.1"] = (
        "Llama-2-7b-chat-hf"
    )
    dtype: str = "float16"
    device: str = "cuda"
    cache_dir: str = "cache/sent_scores"
    generated_responses_dir: str = "cache/generated_responses"

    num_return_sequences: int = 15
    temperature: float = 1.0
    max_new_tokens: int = 512

    """
    Initializes the CustomSelfCheckNLI class.

    Parameters:
    ----------
    device : str, optional
        The device to run the model on (e.g., 'cuda:0' or 'cpu'). Default is 'cuda:0'.
    aggregation : Literal['min', 'max', 'mean'], optional
        The method to aggregate sentence-level scores. Default is 'mean'.
    scores_save_dir : str, optional
        Directory to save or load the calculated sentence-level scores. Default is 'cache/sent_scores'.
    """

    def __post_init__(self):
        self.nli_model = SelfCheckNLI(device="cuda")

    def fit(self, X: pd.DataFrame, y: pd.Series, *args) -> "CustomSelfCheckNLI":
        """ """
        return self

    def transform(self, X: pd.DataFrame) -> list[list[float]]:
        """
        Calculates or loads precomputed sentence-level NLI scores for each response in the dataset.

        Parameters:
        ----------
        X : Any
            DataFrame containing the input data. Must have columns 'response' and 'generated_responses'.

        Returns:
        -------
        scores : List[List[float]]
            List of lists containing the NLI scores for each sentence in each response.
        """

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

        X["generated_responses"] = generated_responses

        data_hash = get_dataframe_hash(X)

        cachefile_general_name = f"{self.model_name}_{data_hash}"

        sent_scores_cache_name = cachefile_general_name + "_sent_scores.joblib"

        sent_scores = get_sent_scores(
            X,
            self.nli_model,
            model_name=self.model_name,
            cache_name=sent_scores_cache_name,
            cache_dir=self.cache_dir,
        )

        return sent_scores

    @log_execution_time(use_cuda_timing=False)
    def predict_score(self, X: Any) -> np.ndarray[float]:
        """
        Predicts and aggregates NLI scores for each response in the dataset.

        Parameters:
        ----------
        X : Any
            DataFrame containing the input data. Must have columns 'response' and 'generated_responses'.

        Returns:
        -------
        scores : List[float]
            List of aggregated NLI scores for each response.
        """

        # Aggregate the scores based on the specified aggregation method
        if self.aggregation == "max":
            scores = [max(sample) for sample in X]
        elif self.aggregation == "min":
            scores = [min(sample) for sample in X]
        elif self.aggregation == "mean":
            scores = [sum(sample) / len(sample) for sample in X]
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation}")

        return scores
