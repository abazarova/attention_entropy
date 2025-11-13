import typing as tp
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import train_test_split

from .dataset_abc import HallucinationDetectionDataset


@dataclass
class CoQA(HallucinationDetectionDataset):
    """A class to process and manage CoQA dataset."""

    model_name: tp.Literal[
        "Mistral-7B-Instruct-v0.1",
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-chat-hf",
        "Llama-3.1-8B-Instruct",
        "Qwen2.5-7B-Instruct",
    ]
    source_file: str = "data/raw/CoQA/coqa_Llama-2-7b-chat-hf.csv"
    split: str = "default"
    val_size: int | float = 100
    random_state: int = 42

    def split_data(self, df: pd.DataFrame) -> tuple[np.ndarray[int], np.ndarray[int]]:
        """Split."""
        if self.split == "haloscope":
            train_val_indices = np.where(df["split"].isin(["val", "train"]))[0]
            test_indices = np.where(df["split"] == "test")[0]
            return train_val_indices, test_indices

        indices = np.arange(len(df))  # Create an array of integer indices
        #self.val_size = len(indices)
        if self.val_size == len(indices):
            return None, indices
        train_test_indices, val_indices = train_test_split(
            indices, test_size=self.val_size, random_state=self.random_state
        )
        return train_test_indices, val_indices


    def load_data(self) -> pd.DataFrame:
        """Load csv with model hallucinations."""

        return pd.read_csv(self.source_file)

    def process(self) -> tuple[pd.DataFrame, pd.Series, ndarray, ndarray | None]:
        df = self.load_data()

        def insert_context_question(row):
            # Assuming 'prompt' is the template where you want to insert context and question
            new_prompt = row["prompt"].format(row["context"], row["question"])
            return new_prompt

        # add dataset name
        df["name"] = "coqa"
        df.rename(columns={"generated_answer": "response"}, inplace=True)
        if self.model_name in [
            "Mistral-7B-Instruct-v0.1",
            "Llama-2-7b-chat-hf",
            "Llama-2-13b-chat-hf",
        ]:
            df["prompt"] = df["context"] + " Q: " + df["question"] + " A:"
            df["id"] = df.index

            # add special tokens
            df["prompt"] = df["prompt"].apply(lambda x: f"<s>{x}")
            df["response"] = df["response"].apply(lambda x: f"{x}</s>")

        elif self.model_name == "Llama-3.1-8B-Instruct":
            df["prompt"] =  df["context"] + " Q: " + df["question"] + " A:"
            df["prompt"] = df["prompt"].apply(lambda x: f"<|begin_of_text|>{x}")
            df["response"] = df["response"].apply(lambda x: f"{x}")

        elif self.model_name == "Qwen2.5-7B-Instruct":
            df["prompt"] = df["context"] + " Q: " + df["question"] + " A:"
            df["prompt"] = df["prompt"].apply(lambda x: f"<|im_start|>user\n Given the context and the subsequent series of question-answer pairs, answer the last question. \n{x}<|im_end|>\n<|im_start|>assistant")
            df["response"] = df["response"]

        else:
            raise NotImplementedError(
                f"This model is not supported yet: {self.model_name}"
            )
        train_indices, test_indices = self.split_data(df)
        return (
            pd.DataFrame(df[["id", "prompt", "response", "name"]]),
            df["hallucination"].astype(int),
            train_indices,
            test_indices,
        )
