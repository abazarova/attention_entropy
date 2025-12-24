from dataclasses import dataclass
from typing import List, Literal, Optional

import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import math

from ..caching_utils import cache_result, get_dataframe_hash
from ..extract_states import get_generated_responses, get_hidden_states
from ..hallucination_detection_abc import HallucinationDetectionMethod
from ..llm_base import LLMBase
from .utils import (
    EntailmentDeberta,
    cluster_assignment_entropy,
    get_semantic_ids,
    logsumexp_by_id,
    predictive_entropy,
    predictive_entropy_rao,
)


@cache_result(cache_dir="cache/semantic_entropy", message="Get semantic entropy")
def get_semantic_entropy(
    X: pd.DataFrame,
    model: LLMBase,
    model_name: str,
    entailment_model: EntailmentDeberta,
    hash: Optional[str] = None,
    cache_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    llm, tokenizer = model.instantiate_llm()

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

    result = {"semantic_entropy": [], "naive_entropy": [], "ca_entropy": []}
    for prompt, responses in tqdm(
        zip(X["prompt"], X["generated_responses"]), total=len(X["prompt"])
    ):
        semantic_ids = get_semantic_ids(
            responses, model=entailment_model, strict_entailment=False
        )

        log_liks_agg = []

        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        for response in responses:
            inputs = tokenizer(
                f"{response}{eos_token}", add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
            input_ids = torch.cat([prompt_ids, inputs], dim=1).to(model.device)

            target_ids = input_ids.clone()
            target_ids[:, : len(prompt_ids[0])] = -100

            with torch.no_grad():
                log_liks_agg.append(-llm(input_ids, labels=target_ids)["loss"].item())
                
        
            
        samples = responses[10:]
        samples_for_est = responses[:10]      
        
        import math
        
        density_ests = []

        
        for sample in samples:
            likelihood_sum = 0
            semantic_density = 0
            for index, sample_for_est in enumerate(samples_for_est):
                    average_likelihood = math.exp(log_liks_agg[index])

                    semantic_distance = entailment_model(sample, sample_for_est)
                    semantic_density += 0.5*(1.0-semantic_distance)*average_likelihood

                    reverse_semantic_distance = entailment_model(sample_for_est, sample)
                    semantic_density += 0.5*(1.0-reverse_semantic_distance)*average_likelihood
                    likelihood_sum += average_likelihood
            density_ests.append(semantic_density/likelihood_sum)
            
        
        
        
        pe =  - np.array(density_ests).log().sum() / len(density_ests)
                

        result["semantic_entropy"].append(pe)

    return result


@dataclass
class SemanticEntropy(HallucinationDetectionMethod):
    model_name: Literal["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.1"] = (
        "Llama-2-7b-chat-hf"
    )
    dtype: str = "float16"
    device: str = "cuda"
    cache_dir: str = "cache/semantic_entropy"
    generated_responses_dir: str = "cache/generated_responses"

    num_return_sequences: int = 15
    temperature: float = 1.0
    max_new_tokens: int = 512

    def fit(self, X_train: list[float], y_train: list[int], *args):
        """
        Trains the linear probe model on the provided dataset.

        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame containing the training data. Must include a column 'is_hal' for target labels.

        Returns:
        --------
        self : LinearProbe
            Returns the instance of the class with the trained model.
        """
        return self

    def transform(self, X):
        """ """

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

        semantic_entropy_cache_name = (
            cachefile_general_name + "_semantic_entropy.joblib"
        )

        semantic_entropy = get_semantic_entropy(
            X,
            llm,
            model_name=self.model_name,
            entailment_model=EntailmentDeberta(self.device),
            cache_name=semantic_entropy_cache_name,
            cache_dir=self.cache_dir,
        )

        return semantic_entropy["semantic_entropy"]

    def predict_score(self, X) -> List[float]:
        """Placeholder."""
        return X
