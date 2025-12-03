import csv
import multiprocessing as mp
import os
from dataclasses import dataclass
from functools import partial
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from loguru import logger
from numpy.typing import NDArray
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
from torch.nn.functional import softmax
from tqdm import trange

from ..extract_states import get_all_attention_maps, get_token_distributions
from ..hallucination_detection_abc import HallucinationDetectionMethod
from ..llm_base import LLMBase
from .utils import attn_maps_to_condent, load_from_cache, save_to_cache


def process_layer(layer_data, response_len):
    """Process a single layer's attention maps."""
    layer, layer_attn_maps = layer_data
    condent = attn_maps_to_condent(layer_attn_maps[0], response_len)  # (n_heads, response_len)
    layer_output = {}
    for head, condent_values in enumerate(condent):
        layer_output[str(head)] = list(condent_values)
    return layer, layer_output

@dataclass
class CondEntRAUQ(HallucinationDetectionMethod):
    """A class to compute prompt-conditioned entropy of the response.

    Attributes
    ----------
    model_name : Literal["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.1"]
        The name of the LLM used. Choices are 'Llama-2-7b-chat-hf' or
        'Mistral-7B-Instruct-v0.1'.
    dtype : str
        The data type used for the LLM inference, e.g., 'float16', 'float32'. Determines the precision of computations.
    device : str
        The device to run the model on, such as 'cuda' for GPU or 'cpu' for CPU. Default is 'cuda'.
    cache_dir : str
        Directory to save or load precomputed scores. If this directory does not exist, it will be created.
        Default is 'cache/condent'.
    analysis_sites : list[tuple[int, int]]
        List of tuples representing pairs of (layer_index, head_index) that are of interest.
    """

    model_name: Literal["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.1"]
    dtype: str = "float16"
    device: str = "cuda"
    cache_dir: str = "cache/condent/ragtruth_qa"

    analysis_sites: Literal["all"] | list[tuple[int, int]] = "all"

    n_layers: int = 32
    n_heads: int = 32
    n_max: int = 10
    alpha: float = 0.2
    aggregation: Literal["mean"] | Literal["max"] = "mean"

    def __post_init__(self):
        """Post initialization of the class."""
        if self.analysis_sites != "all":
            self.analysis_sites = sorted(self.analysis_sites)
        self.cache_dir = f"{self.cache_dir}/{self.model_name}"
        self.llm_model = None

    def transform(self, X: pd.DataFrame) -> list[NDArray[np.float32]]:
        """Calculate CondEnt for each entry in the DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing 'prompt' and 'response' columns.

        Returns
        -------
        list
            List of CondEnt values for each entry.

        """
        if self.llm_model is None:
            self.llm_model = LLMBase(self.model_name, self.dtype, self.device)
            self.llm_model.llm, self.llm_model.tokenizer = (
                self.llm_model.instantiate_llm()
            )

        scores = [self.calc_confidence_scores(X.iloc[i], self.llm_model) for i in trange(len(X))]
        return scores

    def fit(self, *args, **kwargs) -> "CondEntRAUQ":
        """Return the instance of this class."""
        return self

    def predict_score(self, X: list[NDArray[np.float32]]) -> NDArray[np.float32]:
        """Perform inference.

        Parameters
        ----------
        X : list[list[float]]
            List of lists of CondEnt values from selected model heads for each sample in the dataset.

        Returns
        -------
        List[float]
            List of predicted scores for each input example.

        """
        logger.info("Max token entropy is used as a hallucination score.")
        if self.aggregation == "max":
            confidence_scores = [np.max(scores) for scores in X]
        else:
            confidence_scores = [np.mean(scores) for scores in X]
       
        return np.array(confidence_scores)

    def calc_confidence_scores(self, sample: pd.Series, llm_model: LLMBase) -> NDArray[np.float32]:
        """Calculate confidence scores for each response token of the given sample."""
        sample_id = sample["id"]
        filename = f"{sample_id}.json"
        save_path = Path(self.cache_dir) / filename
        
        response = sample["response"]
        response_ids = llm_model.tokenizer(
            response, add_special_tokens=False, return_tensors="pt"
        )
        response_len = len(response_ids[0])
        try:
            condent_dict = load_from_cache(save_path)
        except FileNotFoundError:
            attention_maps = get_all_attention_maps(
                pd.DataFrame([sample]),
                model=llm_model,
            )

            condent_dict = self.compute_condent(
                attention_maps, response_len=response_len, save_path=save_path
            )

        logits = get_token_distributions(pd.DataFrame([sample]), model=llm_model)[0]
        probas = softmax(logits, dim=-1)
        dict_size = probas.shape[-1]
        tokens_entropy = entropy(probas, axis=-1) / np.log(dict_size) # (reponse_len, )
        
        tokens_condent = [condent_dict[str(layer)][str(head)] for layer, head in self.analysis_sites]
        tokens_condent = np.array(tokens_condent).mean(axis=0)

        confidence_scores = self.alpha * tokens_entropy + (1 - self.alpha) * tokens_condent

        return confidence_scores

    def compute_condent(
        self,
        attention_maps: list[torch.Tensor],
        response_len: int | None,
        save_path: Path
    ) -> dict[str, dict[str, list[float]]]:
        """Either load or compute CondEnt values given the attention maps."""
        outputs = {}

        layer_data = list(enumerate(attention_maps))
        process_func = partial(process_layer, response_len=response_len)

        results = [process_func(layer_item) for layer_item in layer_data]
        # Reconstruct outputs dictionary
        for layer, layer_output in results:
            outputs[str(layer)] = layer_output
        save_to_cache(outputs, save_path)

        return outputs
    
    def calc_condent(self, sample: pd.Series, llm_model: LLMBase) -> list[float]:
        """Calculate CondEnt values for the given sample."""
        sample_id = sample["id"]
        filename = f"{sample_id}.json"
        save_path = Path(self.cache_dir) / filename

        try:
            condent_dict = load_from_cache(save_path)
        except FileNotFoundError:
            response = sample["response"]
            response_ids = llm_model.tokenizer(
                response, add_special_tokens=False, return_tensors="pt"
            )
            response_len = len(response_ids[0])

            attention_maps = get_all_attention_maps(
                pd.DataFrame([sample]),
                model=llm_model,
            )

            condent_dict = self.compute_condent(
                attention_maps, response_len=response_len, save_path=save_path
            )

        condent_list = []
        for layer, head in self.analysis_sites:
            condent_list.append(np.mean(condent_dict[str(layer)][str(head)]))

        return condent_list

    def fit_hyperparameters(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Select optimal head subset given the probe dataset."""
        self.analysis_sites = list(product(range(self.n_layers), range(self.n_heads)))
        if self.llm_model is None:
            self.llm_model = LLMBase(self.model_name, self.dtype, self.device)
            self.llm_model.llm, self.llm_model.tokenizer = (
                self.llm_model.instantiate_llm()
            )

        features = [
            self.calc_condent(X_val.iloc[i], self.llm_model) for i in trange(len(X_val))
        ]
        features = np.array(features)  # (n_samples, n_layers * n_heads)

        columns = [f"{i}_{j}" for i, j in self.analysis_sites]
        df = pd.DataFrame(features, columns=columns)

        df["is_hal"] = y_val.values

        avg_distances = pd.DataFrame(
            columns=np.arange(self.n_heads), index=np.arange(self.n_layers)
        )
        hallucinated_scores = df.loc[:, columns][df["is_hal"] == 1].apply(
            np.mean, axis=0
        )
        grounded_scores = df.loc[:, columns][df["is_hal"] == 0].apply(np.mean, axis=0)
        for l, h in self.analysis_sites:
            avg_distances.at[l, h] = (
                hallucinated_scores[f"{l}_{h}"] - grounded_scores[f"{l}_{h}"]
            )
        dist_copy = avg_distances.copy()
        optimal_subset = []
        best_auroc, n_opt = 0, 0
        for n in range(1, self.n_max + 1):
            best_pos = np.unravel_index(np.argmax(dist_copy), dist_copy.shape)  # (h, l)
            optimal_subset.append(best_pos)
            dist_copy[best_pos[1]][best_pos[0]] = -1
            predictions = df[
                [f"{layer}_{head}" for layer, head in optimal_subset]
            ].mean(axis=1)
            roc_auc = roc_auc_score(y_val, predictions)
            if roc_auc > best_auroc:
                n_opt = n
                best_auroc = roc_auc
        self.analysis_sites = optimal_subset[:n_opt]

        # Define the CSV filename
        csv_filename = f"results/condent_rauq_{self.model_name}.csv"

        # Check if file exists, create it if not
        if not os.path.exists(csv_filename):
            # Create the directory if needed (handles nested paths)
            os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
            print(f"Creating new CSV file: {csv_filename}")
        else:
            print(f"Appending to existing CSV file: {csv_filename}")

        # Write to CSV file (mode 'a' for append, 'w' for overwrite)
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Check if file is empty to write header
            if os.path.getsize(csv_filename) == 0:
                writer.writerow(['Model Name', 'Heads'])
            
            # Write data
            writer.writerow([self.model_name, self.analysis_sites])

        print(f"Analysis sites written to {csv_filename}")
