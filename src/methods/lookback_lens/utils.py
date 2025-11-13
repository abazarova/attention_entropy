import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

from src.methods.llm_base import LLMBase


def chspan2tokenspan(
    hallu_spans: list[dict], prompt_len: int, answer_ids: torch.Tensor
):
    token_spans = []
    for span in hallu_spans:
        start, end = span["start"], span["end"]
        tspan = [None, None]
        for i, token_span in enumerate(answer_ids["offset_mapping"][0]):
            if token_span[0] <= start < token_span[1]:
                tspan[0] = i
            if token_span[0] <= end < token_span[1]:
                tspan[1] = i
            if (tspan[0] is not None) and (tspan[1] is not None):
                break
        token_spans.append(tspan)
    return token_spans


def calc_lookback_ratio(attn_maps: torch.Tensor, prompt_len: int, response_len: int):
    for layer_outputs in attn_maps:
        lookback_ratio = []
        layer_outputs = layer_outputs.cpu()
        for i in range(prompt_len, prompt_len + response_len):
            attn_on_context = layer_outputs[0][:, i, :prompt_len].mean(
                -1
            )  # (n_heads, )
            attn_on_generation = layer_outputs[0][:, i, prompt_len : i + 1].mean(
                -1
            )  # (n_heads, )

            lr = attn_on_context / (attn_on_context + attn_on_generation)
            lookback_ratio.append(lr)  # (n_heads, )

        yield torch.stack(lookback_ratio)  # (response_len, n_heads)


def get_lookback_features(
    X: pd.DataFrame,
    y: Optional[list[int]] = None,
    llm_model: LLMBase = None,
    sliding_window: int = 8,
    spans_available: bool = False,
    cache_dir: Path = Path("cache/lookback/Llama-2-7b-chat-hf")
):
    data, labels = [], []
    path_with_spans = cache_dir / f"with_spans/window_{sliding_window}"
    path_no_spans = cache_dir / f"no_spans/window_{sliding_window}"

    path_with_spans.mkdir(parents=True, exist_ok=True)
    path_no_spans.mkdir(parents=True, exist_ok=True)
    if spans_available:
        for (output, prompt_ids, answer_ids), hallu_spans, id_ in tqdm(
            zip(
                llm_model.generate_llm_outputs(
                    X, output_hidden_states=False, output_attentions=True
                ),
                X["labels"],
                X["id"]
            ),
            total=len(X),
        ):
            fname = path_with_spans / f"{id_}.pt"
            if fname.is_file():
                sample_features = torch.load(fname)
                data.extend(sample_features[0])
                labels.extend(sample_features[1])
            else:
                if not hallu_spans:
                    continue
                prompt_len = len(prompt_ids["input_ids"][0])
                response_len = len(answer_ids["input_ids"][0])
                lookback_features = torch.concatenate(
                    list(
                        calc_lookback_ratio(output["attentions"], prompt_len, response_len)
                    ),
                    axis=-1,
                )  # (response_len, n_heads * n_layers)
                hallu_token_pos = chspan2tokenspan(hallu_spans, prompt_ids, answer_ids)

                grounded_features = []
                hallu_features = []
                for i, (s, e) in enumerate(hallu_token_pos):
                    if i == 0 and s > 0:
                        grounded_features.append(lookback_features[:s, :])
                    hallu_features.append(lookback_features[s:e, :])
                    if i == len(hallu_token_pos) - 1 and e < response_len - 1:
                        grounded_features.append(lookback_features[e:, :])
                if grounded_features:
                    grounded_features = torch.cat(grounded_features, dim=0)
                    hallu_features = torch.cat(hallu_features, dim=0)
                    data.extend([grounded_features.mean(0), hallu_features.mean(0)])
                    labels.extend([0, 1])
                    torch.save([[grounded_features.mean(0), hallu_features.mean(0)], [0, 1]], fname)
                else:
                    torch.save([[], []], fname)
    else:
        for (i, (output, prompt_ids, answer_ids)), id_ in tqdm(
            zip(
                enumerate(llm_model.generate_llm_outputs(
                    X, output_hidden_states=False, output_attentions=True
                )),
                X["id"]
            ),
            total=len(X),
        ):
            fname = path_no_spans / f"{id_}.pt"
            if fname.is_file():
                sample_features = torch.load(fname)
                if y is not None:
                    data.extend(sample_features[0])
                    labels.extend([y[i]] * len(sample_features[0]))
                else:
                    data.append(sample_features[0])
            else:
                slices = []
                prompt_len = len(prompt_ids["input_ids"][0])
                response_len = len(answer_ids["input_ids"][0])
                lookback_features = torch.concatenate(
                    list(
                        calc_lookback_ratio(output["attentions"], prompt_len, response_len)
                    ),
                    axis=-1,
                )  # (response_len, n_heads * n_layers)
                for j in range(0, response_len, sliding_window):
                    slices.append(lookback_features[j : j + sliding_window].mean(0))
                    if y is not None:
                        labels.append(y[i])
                if y is not None:
                    data.extend(slices)
                else:
                    data.append(slices)
                torch.save([slices], fname)
    return data, labels
