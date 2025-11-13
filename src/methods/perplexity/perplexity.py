from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..caching_utils import cache_result, get_dataframe_hash
from ..hallucination_detection_abc import HallucinationDetectionMethod
from ..llm_base import LLMBase

nltk.download("punkt")


@cache_result(cache_dir="cache/sent_scores", message="Get sentence scores")
def get_perplexity_scores(
    logits: torch.Tensor,
    answer_ids: torch.Tensor,
    min_k: Optional[float] = None,
    hash: Optional[str] = None,
    cache_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[float]:
    """Compute perplexity with debugging for numerical issues."""
    perplexity_scores = []

    for i, (logit, answer_id) in enumerate(zip(logits, answer_ids)):
        answer_id = answer_id.unsqueeze(-1)
        logit = torch.gather(logit, dim=-1, index=answer_id)
        logit = logit.squeeze(-1)

        perplexity_scores.append(-logit.mean().float())

    return perplexity_scores


@cache_result(cache_dir="cache/token_distribution", message="Get token distributions")
def get_logits(
    X: pd.DataFrame,
    model: LLMBase,
    hash: Optional[str] = None,
    cache_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> list[torch.Tensor]:
    """Retrieve the distribution on each token of the model for each input in the DataFrame."""
    logits = []
    token_ids = []

    for output, prompt_ids, answer_ids in model.generate_llm_outputs(
        X, output_hidden_states=False, output_attentions=False
    ):
        len_answer = len(answer_ids[0])
        logits.append(output.logits[0, -len_answer:].cpu())
        token_ids.append(answer_ids["input_ids"].squeeze(0))

    return logits, token_ids


@dataclass
class Perplexity(HallucinationDetectionMethod):
    model_name: Literal["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.1"] = (
        "Llama-2-7b-chat-hf"
    )

    min_k: Optional[float] = None
    dtype: str = "float16"
    device: str = "cuda"
    cache_dir: str = "cache/perplexity_scores"
    scores_save_dir: str = "cache/generated_logits"

    """
    """

    def __post_init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series, *args) -> "Perplexity":
        """ """
        return self

    def transform(self, X: Any) -> List[List[float]]:
        """ """

        llm = LLMBase(self.model_name, self.dtype, self.device)

        data_hash = get_dataframe_hash(X)

        cachefile_general_name = f"{self.model_name}_{data_hash}"

        generated_logits_cache_name = cachefile_general_name + "_logits.joblib"
        logits, answer_ids = get_logits(
            X, llm, cache_name=generated_logits_cache_name, cache_dir=self.cache_dir
        )

        sent_scores_cache_name = cachefile_general_name + "_perplexity.joblib"
        scores = get_perplexity_scores(
            logits,
            answer_ids,
            self.min_k,
            cache_name=sent_scores_cache_name,
            cache_dir=self.cache_dir,
        )

        return scores

    def predict_score(self, X: Any) -> np.ndarray[float]:
        """"""
        return np.array(X)
