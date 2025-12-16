import numpy as np
import torch

from ..llm_base import move_causal_lm_output_to_cpu
from .rtd_lite import RTD_Lite_summ_only


def get_hidden_states(
    prompt: str,
    response: str,
    layer: int,
    llm,
    tokenizer,
    device: str = "cuda",
    move_to_cpu: bool = True,
):
    prompt_ids: torch.Tensor = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
        return_offsets_mapping=True,
    )["input_ids"]
    answer_ids: torch.Tensor = tokenizer(
        response,
        add_special_tokens=False,
        return_tensors="pt",
        return_offsets_mapping=True,
    )["input_ids"]
    input_ids = torch.cat((prompt_ids, answer_ids), axis=1).to(device)
    len_answer = len(answer_ids[0])
    # Yield the output of the model for the current example
    with torch.no_grad():
        output = llm(
            input_ids,
            output_hidden_states=True,
            output_attentions=False,
        )
    if move_to_cpu:
        output = move_causal_lm_output_to_cpu(output)
        
    with open('/home/llm-factuality/attention_entropy/log.txt', 'w') as f:
        f.write(str(len(output["hidden_states"])))
    return output["hidden_states"][layer][0]


def topology_matrix(hidden_states, network_density: float = 1.0) -> torch.Tensor:
    if len(hidden_states) == 1:
        corr = hidden_states.T @ hidden_states
    else:
        corr = torch.corrcoef(hidden_states.T)
    if network_density < 1.0:
        corr = corr.numpy()
        percentile_threshold = network_density * 100
        threshold = np.percentile(np.abs(corr), 100 - percentile_threshold)
        corr[np.abs(corr) < threshold] = 0
        np.fill_diagonal(corr, 1.0)
        corr = torch.from_numpy(corr)
    adjacency_map = (1 - corr) / 2
    return adjacency_map


def rtd_consistency(base_graph: torch.Tensor, graphs: list[torch.Tensor]):
    scores = []
    for gen_graph in graphs:
        scores.append(RTD_Lite_summ_only(base_graph, gen_graph, dist=True)().item())

    return scores


import functools
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def rtd_consistency_parallel(
    base_graph: torch.Tensor, graphs: list[torch.Tensor], max_workers: int = None
):
    worker_func = functools.partial(RTD_Lite_summ_only, base_graph, dist=True)

    # Use ProcessPoolExecutor for CPU-bound tasks
    # Use ThreadPoolExecutor for I/O-bound tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_func, graph) for graph in graphs]
        scores = [future.result().item() for future in futures]

    return scores
