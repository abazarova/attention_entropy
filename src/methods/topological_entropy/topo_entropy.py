import time
from dataclasses import dataclass
from typing import List, Literal, Optional
import umap
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from ..caching_utils import cache_result, get_dataframe_hash
from ..extract_states import get_generated_responses
from ..hallucination_detection_abc import HallucinationDetectionMethod
from ..llm_base import LLMBase
from .utils import (
    get_hidden_states,
    rtd_consistency,
    topology_matrix,
)

LAYER = 20


@cache_result(cache_dir="cache/topological_entropy", message="Get topo entropy")
def get_topological_consistency(
    X: pd.DataFrame,
    model: LLMBase,
    model_name: str,
    hash: Optional[str] = None,
    cache_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    llm, tokenizer = model.instantiate_llm()

    result = {"topo_entropy": []}
    for prompt, base_response, generated_responses, context, question in tqdm(
        zip(X["prompt"], X["response"], X["generated_responses"], X['context'], X['question']),
        total=len(X["prompt"]),
    ):
        

        N_splits_of_context = 5
        rtd_scores = []
        
        for i in range(N_splits_of_context):
            cur_prompt =  context[len(context) * i / N_splits_of_context  : len(context) * (i + 1)  / N_splits_of_context] + " Q: " + question + " A:"
            cur_prompt =  lambda x: f"<|begin_of_text|>{cur_prompt}"
            hidden_states = get_hidden_states(
                cur_prompt, base_response, LAYER, llm, tokenizer, model.device
            )
            
            dim_method_name = 'pca'
            if dim_method_name == 'pca':
                dim_method = PCA(n_components=min(128, len(hidden_states)))
            elif dim_method_name == 'umap':
                dim_method = umap.UMAP(n_components=min(128, len(hidden_states) - 2))
            elif dim_method_name == 'tsne':
                dim_method = TSNE(n_components=min(128, len(hidden_states)))
                
                    
                
            
            

            hidden_states = dim_method.fit_transform(hidden_states.float())
            base_graph = topology_matrix(torch.from_numpy(hidden_states))
            #base_graph = topology_matrix(hidden_states)
            graphs = []

            for response in generated_responses[:5]:
                

                hidden_states = get_hidden_states(
                    cur_prompt, response, LAYER, llm, tokenizer, model.device
                )
                hidden_states = dim_method.transform(hidden_states.float())
                topology_map = topology_matrix(torch.from_numpy(hidden_states))
                #topology_map = topology_matrix(hidden_states)

                if topology_map.isnan().any():
                    print("NaNs encountered in topology map")
                    breakpoint()
                graphs.append(topology_map)

            rtd_score = rtd_consistency(base_graph, graphs)
            rtd_scores.append(rtd_score)
        final_rtd_score = []  
        for i in range(len(rtd_scores[0])):
            final_rtd_score.append(min([rtd_scores[split][i] for split in range(N_splits_of_context)]))
            
            
        if np.isnan(rtd_score).any():
            print("NaNs encountered in rtd scores")
            breakpoint()
        result["topo_entropy"].append(
            [elem / max(hidden_states.shape[1], 1) for elem in final_rtd_score]
        )

    return result


@dataclass
class TopologicalEntropy(HallucinationDetectionMethod):
    model_name: Literal["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.1"] = (
        "Llama-2-7b-chat-hf"
    )
    dtype: str = "float16"
    device: str = "cuda"
    cache_dir: str = "cache/topological_entropy"
    generated_responses_dir: str = "cache/generated_responses"
    aggregation: str = "mean"

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

        data_hash = get_dataframe_hash(X.drop(['question', 'context'], axis=1))

        cachefile_general_name = f"{self.model_name}_{data_hash}"

        generated_responses_cache_name = (
            cachefile_general_name + "_generated_responses.joblib"
        )

        self.generated_responses_dir = 'cache/generated_responses'

        generated_responses = get_generated_responses(
            X,
            llm,
            cache_name=generated_responses_cache_name,
            cache_dir=self.generated_responses_dir,
        )

        X["generated_responses"] = generated_responses

        data_hash = get_dataframe_hash(X)
        

        cachefile_general_name = f"{self.model_name}_{data_hash}"

        topo_entropy_cache_name = cachefile_general_name + "_topological_entropy_query_splitted_context_psa.joblib"

        topo_entropy = get_topological_consistency(
            X,
            llm,
            model_name=self.model_name,
            cache_name=topo_entropy_cache_name,
            cache_dir=self.cache_dir,
        )
        
        return topo_entropy["topo_entropy"]

    def predict_score(self, X) -> List[float]:
        """Placeholder."""
        if self.aggregation == "mean":
            return [np.mean(elem) for elem in X]
        raise NotImplementedError
