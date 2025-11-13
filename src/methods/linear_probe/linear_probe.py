import os
from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import trange

from ..caching_utils import get_dataframe_hash
from ..customtypes import ModelName
from ..extract_states import get_hidden_states
from ..hallucination_detection_abc import HallucinationDetectionMethod
from ..llm_base import LLMBase
from .dataset import LinearProbeDataset
from .module import LinearProbeModule
from .training import train_model


@dataclass
class LinearProbe(HallucinationDetectionMethod):
    embedding_dim: int = 4096
    layer: int = 16
    lr: float = 1e-3
    batch_size: int = 16
    epochs: int = 10
    validation_size: float = 0.2
    random_state: int = 42
    model_name: Literal["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.1"] = (
        "Llama-2-7b-chat-hf"
    )
    dtype: str = "float16"
    device: str = "cuda"
    cache_dir: str = "cache/hiddens"

    def __post_init__(self):
        """Post initialization."""
        self.module = LinearProbeModule(self.embedding_dim, 2)

    def reset(self):
        self.module = LinearProbeModule(self.embedding_dim, 2)

    def transform(self, X: pd.DataFrame) -> list[torch.Tensor]:
        """Get hidden states and topological features for specified parameters in config."""
        llm_model = LLMBase(
            self.model_name, self.dtype, self.device
        )  # model is initialized in get_features method becauce it needs to be garbage collected after extracting internal states of llm to release memory and GIL

        data_hash = get_dataframe_hash(X)

        cachefile_generall_name = f"{self.model_name}_{data_hash}_{self.layer}"

        hiddens_cache_name = cachefile_generall_name + "_hiddens.joblib"

        hiddens = get_hidden_states(
            X,
            self.layer,
            llm_model,
            cache_name=hiddens_cache_name,
            cache_dir=self.cache_dir,
        )

        del llm_model

        return hiddens

    def fit(
        self,
        X_train: list[torch.Tensor],
        y_train: list[int],
        X_val: list[torch.Tensor],
        y_val: list[int],
    ) -> "LinearProbe":
        train_ds = LinearProbeDataset(X_train, y_train)
        valid_ds = LinearProbeDataset(X_val, y_val)

        train_dataloader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=train_ds.collate_fn,
        )
        valid_dataloader = DataLoader(
            valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=valid_ds.collate_fn,
        )

        optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        self.module, loss_history, accuracy_history = train_model(
            self.module,
            optimizer,
            {"Train": train_dataloader, "Valid": valid_dataloader},
            num_epochs=self.epochs,
            device=self.device,
            save_path=f"{self.cache_dir}/module_layer_{self.layer}.pth",
        )

        return self

    def predict_score(self, X: list[torch.Tensor]) -> np.ndarray[float]:
        """
        Predicts probabilities using the trained linear probe model.

        Parameters:
        ----------
        X : pd.DataFrame
            DataFrame containing the input data.

        Returns:
        -------
        List[float]
            List of predicted probabilities for each input example.
        """

        ds = LinearProbeDataset(X, torch.zeros(len(X)))
        dataloader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, collate_fn=ds.collate_fn
        )

        self.module.eval()
        probs = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["hidden"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits = self.module(input_ids, attention_mask=attention_mask)
                probs_cur = torch.softmax(logits, dim=-1)[:, 1]
                probs.extend(probs_cur.cpu().tolist())

        return np.array(probs)


@dataclass
class LinearProbeSimple(LinearProbe):
    aggregation: Literal["min", "max", "mean", "last"] = "last"

    def __post_init__(self):
        """Post initialization."""
        self.logreg = LogisticRegression(max_iter=1000)

    def reset(self):
        self.logreg = LogisticRegression(max_iter=1000)

    def aggregate(self, hiddens):
        if self.aggregation == "mean":
            return hiddens.mean(dim=0)
        if self.aggregation == "min":
            return hiddens.min(dim=0).values
        if self.aggregation == "max":
            return hiddens.max(dim=0).values
        if self.aggregation == "last":
            return hiddens[-1]

        raise ValueError(f"Unsupported aggregation method: {self.aggregation}")

    def fit(self, X: list[torch.Tensor], y: list[int], *args) -> "LinearProbe":
        """Train linear probe model on the provided dataset.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing the training data.
        y : pd.Series
            Series containing the target labels.

        Returns
        -------
        self : LinearProbe
            Returns the instance of the class with the trained model.

        """
        hiddens_agg = self.prepare_feature_matrix(X)
        self.logreg.fit(hiddens_agg, y)

        return self

    def prepare_feature_matrix(self, X: list[torch.Tensor]) -> np.array:
        hiddens_agg = []
        for h in X:
            hiddens_agg.append(self.aggregate(h))

        hiddens_agg = torch.stack(hiddens_agg, dim=0).numpy()
        return hiddens_agg

    def predict_score(self, X: pd.DataFrame) -> np.ndarray[float]:
        """Predict probabilities using the trained linear probe model.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing the input data.

        Returns
        -------
        List[float]
            List of predicted probabilities for each input example.

        """
        hiddens_agg = self.prepare_feature_matrix(X)
        probs = self.logreg.predict_proba(hiddens_agg)[:, 1]

        return probs
