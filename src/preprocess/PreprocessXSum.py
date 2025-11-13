import typing as tp
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from .dataset_abc import HallucinationDetectionDataset


@dataclass
class XSum(HallucinationDetectionDataset):
    """A class to process and manage CoQA dataset."""

    model_name: tp.Literal[
        "Mistral-7B-Instruct-v0.1",
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-chat-hf",
        "Llama-3.1-8B-Instruct",
        "Qwen2.5-7B-Instruct",
    ]
    source_file: str = "data/raw/XSum/xsum_Llama-2-7b-chat-hf.csv"
    split: str = "original"
    val_size: int | float = 100
    random_state: int = 42

    def split_data(self, df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray]:
        """Split."""
        indices = np.arange(len(df))  # Create an array of integer indices
        # self.val_size = len(indices)
        if self.val_size >= len(indices):
            return None, indices
        train_test_indices, val_indices = train_test_split(
            indices, test_size=self.val_size, random_state=self.random_state
        )
        return train_test_indices, val_indices

    def load_data(self) -> pd.DataFrame:
        """Load csv with model hallucinations."""

        return pd.read_csv(self.source_file)

    def process(self) -> tuple[pd.DataFrame, pd.Series, ndarray | None, ndarray]:
        df = self.load_data()
        df["name"] = "xsum"
        df.rename(
            columns={"generated_summary": "response", "gpt_label": "hallucination"},
            inplace=True,
        )
        train_indices, test_indices = self.split_data(df)
        return (
            pd.DataFrame(df[["id", "prompt", "response", "name"]]),
            df["hallucination"].astype(int),
            train_indices,
            test_indices,
        )
