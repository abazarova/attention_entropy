import typing as tp
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import train_test_split

from .dataset_abc import HallucinationDetectionDataset


@dataclass
class TriviaQA(HallucinationDetectionDataset):
    """A class to process and manage Trivia QA dataset."""

    model_name: tp.Literal[
        "Mistral-7B-Instruct-v0.1",
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-chat-hf",
        "Llama-3.1-8B-Instruct",
        "Meta-Llama-3-8B-Instruct",
    ]
    source_dir: str = "data/raw/SQuAD"
    val_size: int | float = 100
    random_state: int = 42

    def split_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Split."""

        indices = np.arange(len(df))  # Create an array of integer indices
        train_test_indices, val_indices = train_test_split(
            indices, test_size=self.val_size, random_state=self.random_state
        )
        return train_test_indices, val_indices

    def load_data(self) -> pd.DataFrame:
        """Load csv with model hallucinations."""

        return pd.read_csv(f"{self.source_dir}/trivia_{self.model_name}.csv")  ## FIX

    def process(self) -> tuple[pd.DataFrame, pd.Series, ndarray, ndarray | None]:
        df = self.load_data()

        # add dataset name
        df["name"] = "trivia"

        df.rename(columns={"generated_answer": "response"}, inplace=True)
        df["prompt"] = df["prompt"].apply(
            lambda x: "<s>" + x.format(df["context"], df["question"])
        )
        df["id"] = df.index
        df["response"] = df["response"].apply(lambda x: f"{x}</s>")

        # logger.debug(df.columns)
        train_indices, test_indices = self.split_data(df)

        return (
            pd.DataFrame(df[["id", "prompt", "response", "name"]]),
            df["hallucination"].astype(int),
            train_indices,
            test_indices,
        )
