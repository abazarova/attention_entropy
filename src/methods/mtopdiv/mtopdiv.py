from collections import defaultdict
from dataclasses import dataclass
from itertools import chain, combinations, product
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from loguru import logger
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import trange

from ..caching_utils import get_dataframe_hash
from ..extract_states import get_attention_maps
from ..hallucination_detection_abc import HallucinationDetectionMethod
from ..llm_base import LLMBase
from .utils import (
    attn_mx_to_mtopdiv,
    load_from_cache,
    save_to_cache,
    transform_attention_scores_to_distances,
)


def select_best_features(
    X: pd.DataFrame, y: pd.Series, n_features: int = 10, method: str = "f_classif"
) -> list:
    """
    Select best features using univariate statistical tests

    Args:
        X: DataFrame with features (columns are feature names)
        y: Series with true labels
        n_features: Number of top features to select
        method: 'f_classif' (ANOVA F-value) or 'mutual_info_classif' (mutual information)

    Returns:
        List of selected feature names
    """
    # Validate inputs
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    if not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")

    # Choose scoring method
    if method == "f_classif":
        scorer = f_classif
    elif method == "mutual_info_classif":
        scorer = mutual_info_classif
    else:
        raise ValueError("method must be 'f_classif' or 'mutual_info_classif'")

    # Select top features
    selector = SelectKBest(score_func=scorer, k=min(n_features, X.shape[1]))
    selector.fit(X, y)

    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()

    return selected_features


# logger.disable("src.methods.caching_utils")


@dataclass
class MTopDiv(HallucinationDetectionMethod):
    """A class to compute MTopDivergence between tokens of a prompt and a response.

    This class provides methods to calculate MTopDivergence (MTopDiv) for given text prompts and responses using
    pre-trained language models.

    Attributes
    ----------
    model_name : Literal["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.1"]
        The name of the pre-trained model to use for computing divergences. Choices are 'Llama-2-7b-chat-hf' or
        'Mistral-7B-Instruct-v0.1'.
    dtype : str
        The data type used for the LLM inference, e.g., 'float16', 'float32'. Determines the precision of computations.
    device : str
        The device to run the model on, such as 'cuda' for GPU or 'cpu' for CPU. Default is 'cuda'.
    cache_dir : str
        Directory to save or load precomputed MTopDiv data. If this directory does not exist, it will be created.
        Default is 'cache/mtopdiv'.

    mode : Literal["supervised", "unsupervised"]
        Specifies whether to evaluate the obtained results in a supervised or unsupervised mode.
    analysis_sites : list[tuple[int, int]]
        List of tuples representing pairs of (layer_index, head_index) that are of interest.
    zero_out : Literal["prompt", "response"]
        Determines whether to zero out distances between prompt tokens or response tokens.
    normalize_by_length : bool
        Indicates whether to divide the obtained MTopDiv values by the length of the response or prompt
        (depending on which is zeroed out). Default is `False`.

    """

    model_name: Literal["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.1"]
    dtype: str = "float16"
    device: str = "cuda"
    cache_dir: str = "cache/mtopdiv/ragtruth_qa"

    mode: Literal["supervised", "unsupervised"] = "supervised"
    analysis_sites: Union[Literal["all"], list[tuple[int, int]]] = "all"
    zero_out: Literal["prompt", "response"] = "prompt"
    normalize_by_length: bool = True

    n_jobs: int = 16
    critical_size: int = 768
    n_layers: int = 32
    n_heads: int = 32
    n_max: int = 6  # hyperparameter

    def __post_init__(self):
        """Post initialization of the class."""
        if self.analysis_sites != "all":
            self.analysis_sites = sorted(self.analysis_sites)
        self.clf = None
        self.cache_dir = (
            Path(self.cache_dir) / f"zero_out_{self.zero_out}" / (self.model_name)
        )
        self.llm_model = None

    def transform(self, X: pd.DataFrame) -> list[list[float]]:
        """Calculate MTopDiv for each entry in the DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing 'prompt' and 'response' columns.

        Returns
        -------
        list
            List of MTopDiv values for each entry.

        """
        if self.llm_model is None:
            self.llm_model = LLMBase(self.model_name, self.dtype, self.device)
            self.llm_model.llm, self.llm_model.tokenizer = (
                self.llm_model.instantiate_llm()
            )

        mtopdiv_list = [
            self.calc_mtopdiv(X.iloc[i], self.llm_model) for i in trange(len(X))
        ]
        return mtopdiv_list

    def fit(self, X_train: list[list[float]], y_train: list[int], *args) -> "MTopDiv":
        """Train the logistic regression on the MTopDiv features for provided dataset.

        Parameters
        ----------
        X_train: list
            List containing the training data.
        y_train : list
            List containing the target labels.

        Returns
        -------
        self : MTopDiv
            Returns the instance of the class with the trained model.

        """
        return self

    def predict_score(self, X: list[list[float]]) -> np.ndarray[float]:
        """Perform inference.

        Parameters
        ----------
        X : list[list[float]]
            List of lists of MTopDiv values from fixed model heads for each sample in the dataset.

        Returns
        -------
        List[float]
            List of predicted probabilities for each input example.

        """
        X = np.array(X)

        if self.mode == "supervised":
            logger.info(
                "MTopDiv-based classifier prediction is used as a hallucination score."
            )
            return self.clf.predict_proba(X)[:, 1]
        elif self.mode == "unsupervised":
            logger.info("Average MTopDiv is used as a hallucination score.")
            return np.abs(X).mean(axis=-1)
        else:
            raise ValueError(
                "This mode is not supported. Please selected either 'supervised' or 'unsupervised' mode."
            )

    def calc_mtopdiv(self, sample: pd.Series, llm_model: LLMBase) -> list[list[float]]:
        """Calculate MTop-Div values for the given sample."""
        sample_id = sample["id"]
        filename = "layer_{}/head_{}/" + f"{sample_id}.json"

        try:
            return [
                load_from_cache(self.cache_dir / filename.format(layer, head))
                for layer, head in self.analysis_sites
            ]
        except FileNotFoundError:
            pass

        sample_hash = get_dataframe_hash(sample)
        cachefile = f"{self.model_name}_{sample_hash}" + "_layer_{}"

        response = sample["response"]
        response_ids = llm_model.tokenizer(
            response, add_special_tokens=False, return_tensors="pt"
        )
        response_len = len(response_ids[0])

        attention_maps = [
            attn_map
            for layer in range(self.n_layers)
            for attn_map in get_attention_maps(
                pd.DataFrame([sample]),
                layer=layer,
                model=llm_model,
                cache_name=cachefile.format(layer),
            )
        ]

        sample_size = attention_maps[0].shape[-1]
        n_jobs = self.n_jobs if sample_size <= self.critical_size else 1
        mtopdiv_list = self.load_or_compute_mtopdiv(
            attention_maps, response_len=response_len, filename=filename, n_jobs=n_jobs
        )
        return mtopdiv_list

    def load_or_compute_mtopdiv(
        self,
        attention_maps: list[torch.Tensor],
        response_len: Optional[int],
        filename: str,
        n_jobs: int,
    ) -> list[float]:
        """Either load or compute MTopDiv values given the attention maps."""
        mtopdiv_array = np.empty(len(self.analysis_sites), dtype=object)
        mtopdiv_array[:] = -1

        mtopdiv_array, missing_positions = (
            self._fill_array_and_identify_missing_positions(mtopdiv_array, filename)
        )
        missing_values = self._compute_missing_values(
            attention_maps, response_len, filename, missing_positions, n_jobs
        )
        mtopdiv_array[mtopdiv_array == -1] = np.concatenate(missing_values)

        return list(mtopdiv_array)

    def _fill_array_and_identify_missing_positions(
        self, mtopdiv_array: np.ndarray, filename: str
    ) -> tuple[np.ndarray, dict]:
        """Fill the array with the values from cache and identify positions that need to be computed."""
        missing_positions = defaultdict(list)

        for i, (i_layer, j_head) in enumerate(self.analysis_sites):
            data_path = self.cache_dir / filename.format(i_layer, j_head)
            if data_path.exists():
                mtopdiv = load_from_cache(data_path)
                mtopdiv_array[i] = mtopdiv
            else:
                missing_positions[i_layer].append(j_head)

        return mtopdiv_array, missing_positions

    def _compute_missing_values(
        self,
        attention_maps: list[torch.Tensor],
        response_len: Optional[int],
        filename: str,
        missing_positions: dict[int, int],
        n_jobs: int,
    ) -> list[list[float]]:
        """Compute missing values and save them to cache."""
        layers = list(missing_positions.keys())

        missing_values = []
        for layer in sorted(layers):
            torch.cuda.empty_cache()

            heads = missing_positions[layer]
            heads = torch.tensor(heads).to(self.device)
            attn_mxs_tmp = torch.index_select(
                attention_maps[layer].to(self.device), 0, heads
            )
            distance_mxs = transform_attention_scores_to_distances(
                attn_mxs_tmp, self.zero_out, response_len
            )
            barcodes, mtopdiv_parallel = zip(
                *Parallel(n_jobs=n_jobs)(
                    delayed(attn_mx_to_mtopdiv)(distance_mx)
                    for distance_mx in distance_mxs
                )
            )
            missing_values.append(mtopdiv_parallel)
            for i, head in enumerate(heads):
                data_path = self.cache_dir / filename.format(layer, head)
                save_to_cache(
                    barcodes[i], float(mtopdiv_parallel[i]), response_len, data_path
                )
        return missing_values

    def fit_hyperparameters(self, X_val: pd.DataFrame, y_val: pd.DataFrame):
        """Select optimal head subset given the probe dataset."""
        self.analysis_sites = product(range(self.n_layers), range(self.n_heads))
        self.analysis_sites = sorted(self.analysis_sites)
        self.llm_model = LLMBase(self.model_name, self.dtype, self.device)
        self.llm_model.llm, self.llm_model.tokenizer = self.llm_model.instantiate_llm()

        features = [
            self.calc_mtopdiv(X_val.iloc[i], self.llm_model) for i in trange(len(X_val))
        ]
        features = np.array(features)  # (n_samples, n_layers * n_heads)

        columns = [f"{i}_{j}" for i, j in self.analysis_sites]
        df = pd.DataFrame(features, columns=columns)

        if self.mode == "supervised":
            best_auc = 0
            for n in range(1, self.n_max + 1):
                # Use Recursive Feature Elimination to select top n features
                selected_features = select_best_features(df, y_val, n)
                # Train model on selected features
                clf_temp = LogisticRegression()
                clf_temp.fit(df[selected_features], y_val)

                # Predict on the same data and calculate ROC AUC
                y_pred_proba = clf_temp.predict_proba(df[selected_features])[:, 1]
                auc_score = roc_auc_score(y_val, y_pred_proba)

                # Check if this is the best performance so far
                if auc_score > best_auc:
                    best_auc = auc_score
                    best_features = selected_features
                    best_clf = clf_temp

            self.clf = best_clf
            f = lambda elem: (int(elem[0]), int(elem[1]))
            self.analysis_sites = [f(elem.split("_")) for elem in best_features]
        elif self.mode == "unsupervised":
            df["is_hal"] = y_val.values

            avg_distances = pd.DataFrame(
                columns=np.arange(self.n_heads), index=np.arange(self.n_layers)
            )
            hallu_mtd = df.loc[:, columns][df["is_hal"] == 1].apply(np.mean, axis=0)
            grnd_mtd = df.loc[:, columns][df["is_hal"] == 0].apply(np.mean, axis=0)
            for l, h in self.analysis_sites:
                avg_distances.at[l, h] = hallu_mtd[f"{l}_{h}"] - grnd_mtd[f"{l}_{h}"]
            dist_copy = avg_distances.copy()

            optimal_subset = []
            best_auroc, n_opt = 0, 0
            for n in range(1, self.n_max + 1):
                best_pos = np.unravel_index(
                    np.argmax(dist_copy), dist_copy.shape
                )  # (h, l)
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
        else:
            raise ValueError(
                "This mode is not supported. Please selected either 'supervised' or 'unsupervised' mode."
            )
        # Append to JSON file
        import json
        import os

        def append_heads_to_json(heads_list, filename="selected_heads.json"):
            # Convert all elements to native Python ints
            heads_list_serializable = []
            for head in heads_list:
                if isinstance(head, (list, tuple)):
                    heads_list_serializable.append([int(h) for h in head])
                else:
                    heads_list_serializable.append(int(head))

            # Initialize data structure
            data = {"runs": []}

            # If file exists and has content, load existing data
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                try:
                    with open(filename, "r") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    print(
                        f"Warning: {filename} contains invalid data. Creating new file."
                    )
                    data = {"runs": []}

            # Append new run
            data["runs"].append({"selected_heads": heads_list_serializable})

            # Write back to file
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)

            print(f"Appended {len(heads_list)} heads to {filename}")

        # Usage
        append_heads_to_json(self.analysis_sites)
        print("SELECTED HEADS:", self.analysis_sites)
