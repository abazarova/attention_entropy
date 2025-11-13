from .llm_utils import forward, wrapper
from transformers.models.llama import modeling_llama
from transformers.models.mistral import modeling_mistral

from typing import List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import gc
import torch.nn.functional as F
from dataclasses import dataclass

from ..hallucination_detection_abc import HallucinationDetectionMethod

from ..llm_base import LLMBase
from ..customtypes import ModelName
from ..caching_utils import get_dataframe_hash, cache_result

def calculate_dist(sep_vocabulary_dist, sep_attention_dist):
    softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)  
    softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)  

    M = 0.5 * (softmax_mature_layer + softmax_anchor_layer) 

    # 4. Calculate log-softmax for the KL divergence
    log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)  
    log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1) 

    # 5. Calculate the KL divergences and then the JS divergences
    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').mean(-1)  
    kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none').mean(-1)  
    # # Fix bug: https://github.com/Jeryi-Sun/ReDEeP-ICLR/issues/2 but for stable calculation, we maintain the original implementation of JSD.
    # kl1 = F.kl_div(M.log(), softmax_mature.unsqueeze(0),  reduction='none').mean(-1)
    # kl2 = F.kl_div(M.log(), softmax_anchor,  reduction='none').mean(-1)
    js_divs = 0.5 * (kl1 + kl2) 
        
    return js_divs*10e5

@cache_result(cache_dir="cache/redeep", message="Get ReDeEP scores")
def compute_redeep_scores(
    X: pd.DataFrame,
    model: LLMBase,
    copy_heads,
    knowledge_ffns,
    hash: Optional[str] = None,
    cache_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> list[torch.Tensor]:
    """"""
    wrapper.reset()
    if model.llm is None:
        model.llm, model.tokenizer = model.instantiate_llm()
        model.llm.config._attn_implementation = "eager"
    wrapper.post_init(model.llm)

    knowledge_diff_batch = []
    external_similarity_batch = []
    
    for output, prompt_ids, answer_ids in tqdm(model.generate_llm_outputs(
        X, output_hidden_states=True, output_attentions=True
    ), total=len(X)):
        knowledge_diff = []
        external_similarity = []

        attns = torch.cat(output.attentions, dim=0)
        layer_idxs, head_idxs = zip(*copy_heads)
        layer_idxs = torch.tensor(layer_idxs)
        head_idxs = torch.tensor(head_idxs)
        selected_attn = attns[layer_idxs, head_idxs]

        for idx in range(len(prompt_ids[0]), len(prompt_ids[0]) + len(answer_ids[0])):
            pointer_scores = selected_attn[:, idx, :]

            pointer_probs = pointer_scores[:,:len(prompt_ids[0])]   

            top_k = int(pointer_probs.shape[-1] * 0.1)  # 10% of sequence length
            sorted_indices = torch.argsort(pointer_probs, dim=1, descending=True)
            top_k_indices = sorted_indices[:, :top_k]

            flattened_indices = top_k_indices.flatten()  # shape (head_num * k,)
            selected_hidden_states = output.hidden_states[-1][0, :, :][flattened_indices]  # shape (head_num * k, hidden_state)
            top_k_hidden_states = selected_hidden_states.view(top_k_indices.shape[0], top_k_indices.shape[1], -1)
            attend_token_hidden_state = torch.mean(top_k_hidden_states, dim=1) # (head_num, hidden_state)

            current_hidden_state = output.hidden_states[-1][0, idx, :] # shape (hidden_state,)
            current_hidden_state = current_hidden_state.unsqueeze(0).expand(attend_token_hidden_state.shape)

            cosine_similarity = F.cosine_similarity(attend_token_hidden_state, current_hidden_state, dim=1)
            
            external_similarity.append(sum(cosine_similarity).cpu().item())

        del (
            output, attns, selected_attn, pointer_scores, pointer_probs, 
            selected_hidden_states, top_k_hidden_states, attend_token_hidden_state, 
            current_hidden_state, cosine_similarity
        )
        
        torch.cuda.empty_cache()
        gc.collect()
        
        for idx in range(len(prompt_ids[0]), len(prompt_ids[0]) + len(answer_ids[0])):
            knowledge_diff.append(sum([
                calculate_dist(
                    mature_logits[0, idx].to(model.device),
                    anchor_logits[0, idx].to(model.device),
                ).cpu().item() for i, (anchor_logits, mature_logits) in enumerate(wrapper) if i in knowledge_ffns
            ]))
        wrapper.reset() 

        knowledge_diff = sum(knowledge_diff) / len(knowledge_diff)
        external_similarity = sum(external_similarity) / len(external_similarity)

        knowledge_diff_batch.append(knowledge_diff)
        external_similarity_batch.append(external_similarity)

    scores = np.stack((knowledge_diff_batch, external_similarity_batch), axis=1)
    return scores


@dataclass
class ReDeEP(HallucinationDetectionMethod):
    """"""

    model_name: ModelName
    dtype: str = "float16"
    device: str = "cuda"
    cache_dir: str = "cache/redeep"

    knowledge_ffns: List[int] = None
    copy_heads: List[Tuple[int, int]] = None
    alpha: float = 1.0
    beta: float = 1.0

    def transform(self, X):
        """"""
        if self.model_name not in ["Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Mistral-7B-Instruct-v0.1"]:
            raise ValueError(f"Unsupported model: {self.model_name}")

        modeling_llama.LlamaDecoderLayer.forward = forward
        modeling_mistral.MistralDecoderLayer.forward = forward
        
        llm_model = LLMBase(self.model_name, self.dtype, self.device)


        data_hash = get_dataframe_hash(X)
        redeep_cache_name = f"{self.model_name}_{data_hash}_redeep.joblib"
        redeep_scores = compute_redeep_scores(
            X,
            llm_model,
            self.copy_heads,
            self.knowledge_ffns,
            cache_name=redeep_cache_name,
            cache_dir=self.cache_dir,
        )

        return redeep_scores

    def fit(self, X, y):
        return self

    def predict_score(self, X: np.array) -> List[float]:
        """"""
        X = np.vstack(X)
        pred = X[:, 0] * self.alpha - X[:, 1] * self.beta
        return pred
