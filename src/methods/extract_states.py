"""Module with functions for extraction of internal states of LLM."""

from typing import Optional

import pandas as pd
import torch

from .caching_utils import cache_result
from .llm_base import LLMBase
from .performance_monitor import log_execution_time


@log_execution_time(use_cuda_timing=True)
@cache_result(cache_dir="cache/hiddens", message="Get hidden states")
def get_hidden_states(
    X: pd.DataFrame,
    layer: int,
    model: LLMBase,
    hash: Optional[str] = None,
    cache_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    move_to_cpu: bool = True,
) -> list[torch.Tensor]:
    """Retrieve the hidden states from the specified layer of the model for each input in the DataFrame.

    This function generates hidden states from a specified LLM layer for each input row in the DataFrame.
    The results are cached for faster retrieval on subsequent runs.

    Args:
        X (pd.DataFrame): DataFrame containing input data with text prompts.
        layer (int): The specific layer of the LLM from which to extract hidden states.
        model (LLMBase): An instance of the LLMBase class to generate the model's outputs.
        hash (Optional[str], optional): Unique hash to identify cache results. Defaults to None.
        cache_name (Optional[str], optional): Custom cache file name. Defaults to None.
        cache_dir (Optional[str], optional): Directory to store cache results. Defaults to None.

    Returns:
        list[torch.Tensor]: A list of hidden state tensors extracted from the specified layer.

    """
    hiddens_list = []

    for output, _, answer_ids in model.generate_llm_outputs(
        X, output_hidden_states=True, output_attentions=False, move_to_cpu=move_to_cpu
    ):
        len_answer = len(answer_ids[0])
        hiddens_list.append(output["hidden_states"][layer][0, -len_answer:].cpu())

    return hiddens_list


# @cache_result(cache_dir="cache/attention_maps", message="Get attention maps")
@log_execution_time(use_cuda_timing=True)
def get_attention_maps(
    X: pd.DataFrame,
    layer: int,
    model: LLMBase,
    hash: Optional[str] = None,
    cache_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> list[torch.Tensor]:
    """Retrieve the attention maps from the specified layer of the model for each input in the DataFrame.

    This function generates attention maps from a specified LLM layer for each input row in the DataFrame.
    The results are cached for faster retrieval on subsequent runs.

    Args:
        X (pd.DataFrame): DataFrame containing input data with text prompts.
        layer (int): The specific layer of the LLM from which to extract attention maps.
        model (LLMBase): An instance of the LLMBase class to generate the model's outputs.
        hash (Optional[str], optional): Unique hash to identify cache results. Defaults to None.
        cache_name (Optional[str], optional): Custom cache file name. Defaults to None.
        cache_dir (Optional[str], optional): Directory to store cache results. Defaults to None.

    Returns:
        list[torch.Tensor]: A list of attention map tensors extracted from the specified layer.

    """
    attention_maps_list = []
    for output, _, _ in model.generate_llm_outputs(
        X, output_hidden_states=False, output_attentions=True
    ):
        all_layers_attention_maps = output["attentions"][layer].squeeze(0).cpu()
        attention_maps_list.append(all_layers_attention_maps)
        del all_layers_attention_maps
        torch.cuda.empty_cache()

    return attention_maps_list


def get_all_attention_maps(X: pd.DataFrame, model: LLMBase) -> list[torch.Tensor]:
    """Retrieve the attention maps from the specified layer of the model for each input in the DataFrame.

    This function generates attention maps from a specified LLM layer for each input row in the DataFrame.
    The results are cached for faster retrieval on subsequent runs.

    Args:
        X (pd.DataFrame): DataFrame containing input data with text prompts.
        layer (int): The specific layer of the LLM from which to extract attention maps.
        model (LLMBase): An instance of the LLMBase class to generate the model's outputs.
        hash (Optional[str], optional): Unique hash to identify cache results. Defaults to None.
        cache_name (Optional[str], optional): Custom cache file name. Defaults to None.
        cache_dir (Optional[str], optional): Directory to store cache results. Defaults to None.

    Returns:
        list[torch.Tensor]: A list of attention map tensors extracted from the specified layer.

    """
    output = next(
        model.generate_llm_outputs(
            X, output_hidden_states=False, output_attentions=True
        )
    )
    attention_maps = output[0]["attentions"]
    attention_maps = [elem.cpu() for elem in attention_maps]
    torch.cuda.empty_cache()

    return attention_maps


# @log_execution_time(use_cuda_timing=True)
# @cache_result(cache_dir="cache/token_distribution", message="Get token distributions")
def get_token_distributions(
    X: pd.DataFrame,
    model: LLMBase,
    hash: Optional[str] = None,
    cache_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> list[torch.Tensor]:
    """Retrieve the distribution on each token of the model for each input in the DataFrame."""
    logits_list = []
    for output, prompt_ids, answer_ids in model.generate_llm_outputs(
        X, output_hidden_states=False, output_attentions=False
    ):
        len_answer = len(answer_ids[0])
        logits_list.append(output.logits[0, -len_answer:].cpu())

    return logits_list


@log_execution_time(use_cuda_timing=True)
@cache_result(cache_dir="cache/generated_responses", message="Get generated responses")
def get_generated_responses(
    X: pd.DataFrame,
    model: LLMBase,
    hash: Optional[str] = None,
    cache_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    generated_responses = []
    for responses in model.generate_llm_responses(X):
        generated_responses.append(responses)

    return generated_responses
