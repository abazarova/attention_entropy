import json
from pathlib import Path
from typing import Literal

import mtd.barcodes as mtd
import numpy as np
import ripserplusplus as rpp_py
import torch


def transform_attention_scores_to_distances(
    attn_mxs: torch.Tensor,
    zero_out: Literal["prompt", "response"],
    len_answer: int,
    lower_bound: float = 0.0,
) -> torch.Tensor:
    """Transform attention matrix to the matrix of distances between tokens.

    Parameters
    ----------
    attn_mxs : torch.Tensor
        Attention matrixes of one sample (n_heads x n_tokens x n_tokens).
    zero_out : Literal['prompt', 'response']
        Determines whether to zero out distances between prompt tokens or response tokens.
    len_answer : int
        Length of the response.

    Returns
    -------
    torch.Tensor
        Distance matrix.

    """
    n_tokens = attn_mxs.shape[1]
    distance_mx = 1 - torch.clamp(
        attn_mxs, min=lower_bound
    )  # torch.where(attn_mx > lower_bound, attn_mx, 0.0)
    zero_diag = torch.ones(n_tokens, n_tokens) - torch.eye(n_tokens)
    distance_mx *= zero_diag.to(attn_mxs.device).expand_as(
        distance_mx
    )  # torch.diag(torch.diag(distance_mx))
    distance_mx = torch.minimum(distance_mx.transpose(1, 2), distance_mx)

    if zero_out == "prompt":
        len_prompt = n_tokens - len_answer
        distance_mx[:, :len_prompt, :len_prompt] = 0
    elif zero_out == "response":
        distance_mx[:, -len_answer:, -len_answer:] = 0
    else:
        raise ValueError(f"Unsupported zero_out parameter: {zero_out}")

    return distance_mx.cpu().float().numpy()


def attn_mx_to_mtopdiv(distance_mx: np.ndarray) -> tuple[np.ndarray, float]:
    """Calculate barcodes and MTopDiv value for the given attention matrix."""
    barcodes = rpp_py.run("--format distance --dim 1", distance_mx)
    barcodes = mtd.barc2array(barcodes)
    mtopdiv = mtd.get_score(barcodes, 0, "sum_length")
    return barcodes, mtopdiv


def save_to_cache(
    barcodes: np.ndarray,
    mtopdiv: float,
    response_len: int,
    data_path: Path,
):
    """Save the barcodes and MTopDiv value to cache."""
    data_path.parent.mkdir(exist_ok=True, parents=True)
    data = {
        "barcodes_0": barcodes[0].tolist(),
        "barcodes_1": barcodes[1].tolist(),
        "mtopdiv": mtopdiv,
        "response_len": response_len,
    }
    with open(data_path, "w") as f:
        json.dump(data, f)


def load_from_cache(path: Path, normalize_by_length: bool = True) -> float:
    """Load the precomputed data from cache."""
    with open(path) as f:
        data = json.load(f)
    mtopdiv = data["mtopdiv"]
    if normalize_by_length:
        mtopdiv /= data["response_len"]
    return mtopdiv
