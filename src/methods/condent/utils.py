import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import entropy


def save_to_cache(
    data: dict[str, dict[str, list[float]]],
    save_path: Path,
):
    """Save the condent values to cache."""
    save_path.parent.mkdir(exist_ok=True, parents=True)
    try:
        with open(save_path, "w") as f:
            json.dump(data, f)
    except:
        raise


def load_from_cache(path: Path) -> dict[str, dict[str, list[float]]]:
    """Load the precomputed data from cache."""
    with open(path) as f:
        data = json.load(f)
    return data


def attn_maps_to_condent(attn_maps: torch.Tensor, response_len: int):
    """Transform attention map to the conditional entropy values.

    Parameters
    ----------
    attn_maps : torch.Tensor
        Attention maps of one sample (n_heads x n_tokens x n_tokens).
    response_len : int
        Length of the response.

    Returns
    -------
    list[float]

    """
    n_tokens = attn_maps.shape[1]
    prompt_len = n_tokens - response_len
    assert prompt_len > 1, "Context too short"
    condent_values = [] # (response_len, n_heads)
    for row_idx in range(prompt_len, n_tokens):
        rows = attn_maps[:, row_idx]

        prompt_rows = rows[:, :prompt_len]
        normalized_prompt_rows = prompt_rows / (prompt_rows.sum(axis=-1, keepdims=True) + 1e-5)
        prompt_entropy = entropy(normalized_prompt_rows.float(), axis=-1)
        prompt_entropy = np.where(np.isnan(prompt_entropy), 1, prompt_entropy)
        condent = prompt_entropy / np.log(prompt_len)
        assert not np.isnan(condent).any(), "NaN encountered in entropy calculation"
        condent_values.append(condent)

    return np.array(condent_values).T # (n_heads, response_len)
