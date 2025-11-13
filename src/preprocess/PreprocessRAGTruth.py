import json
import os
import typing as tp
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from .dataset_abc import HallucinationDetectionDataset


@dataclass
class RAGTruth(HallucinationDetectionDataset):
    """A class to process and manage RAGTruth dataset.

    Attributes
    ----------
    model : str
        The name of the model to be used for processing.
    keep_columns : list of str
        Columns to keep in the final saved dataframe.
    task_type : str
        The type of task (e.g., QA) to filter the data on.
    source_dir : str
        Directory path for raw data files.
    save_dir : str
        Directory path to save processed data.
    split : str
        Method to split the data (either 'original' or 'train_test_split').
    val_size : float
        Proportion of the dataset to include in the probe split when using 'train_test_split'.
    random_state : int
        Random seed for data splitting when using 'train_test_split'.

    """

    model_name: Literal[
        "Mistral-7B-Instruct-v0.1", "Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf"
    ]
    #  keep_columns: list[str] = ['prompt', 'temperature', 'split', 'response', 'is_hal'],
    task_type: str = "QA"
    source_dir: str = "data/raw/RAGTruth"
    save_dir: str = "data/processed/RAGTruth"
    val_size: int | float = 100
    random_state: int = 42
    split: str = "default"

    names_dict: dict = field(
        default_factory=lambda: {
            "Mistral-7B-Instruct-v0.1": "mistral-7B-instruct",
            "Llama-2-7b-chat-hf": "llama-2-7b-chat",
            "Llama-2-13b-chat-hf": "llama-2-13b-chat",
        }
    )

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads the raw response and source data from JSONL files.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Two dataframes: one for responses and one for source information.

        """  # noqa: D401
        try:
            responses = pd.read_json(
                path_or_buf=f"{self.source_dir}/response_new.jsonl", lines=True
            )
            source_info = pd.read_json(
                path_or_buf=f"{self.source_dir}/source_info_new.jsonl", lines=True
            )
            return responses, source_info

        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading data from {self.source_dir}: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def filter_and_label_data(
        self, source_df: pd.DataFrame, responses_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merges, filters, and labels the data based on the specified task type and model.

        Parameters
        ----------
        source_df : pd.DataFrame
            The dataframe containing source information.
        responses_df : pd.DataFrame
            The dataframe containing response data.

        Returns
        -------
        pd.DataFrame
            The filtered and labeled dataframe.

        """  # noqa: D401
        model_n = self.names_dict[self.model_name]

        # Merge source and response data on 'source_id'
        df = pd.merge(source_df, responses_df, how="inner", on="source_id")

        # Filter by task type and model, and create 'is_hal' label
        if self.task_type == "All":
            qa = df[
                (df["task_type"] == "QA")
                & (df["model"] == model_n)
                & (df["quality"] == "good")
            ].copy()

            summ = df[
                (df["task_type"] == "Summary")
                & (df["model"] == model_n)
                & (df["quality"] == "good")
            ].copy()

            if self.model_name == "Llama-2-13b-chat-hf":
                summ["length"] = summ["response"].apply(
                    lambda x: len(x.split(" "))
                ) + summ["prompt"].apply(lambda x: len(x.split(" ")))
                summ = summ[summ["length"] <= 1450]
                summ.sort_values(by=["length"], ascending=False, inplace=True)
                summ.drop(columns=["length"], inplace=True)

            with open(f"/app/common_ids_{self.model_name}.json") as f:
                data2txt_ids = json.load(f)
            print(f"Processed {len(data2txt_ids)} successfully")
            data2txt = df[
                (df["task_type"] == "Data2txt")
                & (df["model"] == model_n)
                & (df["quality"] == "good")
                & (df["id"].isin(data2txt_ids))
            ].copy()

            # data2txt["length"] = data2txt["response"].apply(
            #     lambda x: len(x.split(" "))
            # ) + data2txt["prompt"].apply(lambda x: len(x.split(" ")))

            # cnt_grounded = len(data2txt[data2txt["is_hal"] == 0])
            # keep_hallu = cnt_grounded * 2
            # ids_to_keep = (
            #     data2txt[data2txt["is_hal"] == 0]["id"].tolist()
            #     + data2txt[data2txt["is_hal"] == 1]["id"].sample(keep_hallu).tolist()
            # )
            # data2txt = data2txt[data2txt["id"].isin(ids_to_keep)]
            # data2txt = data2txt[data2txt["length"] <= 1024]
            # data2txt.drop(columns=["length"], inplace=True)

            tmp = pd.concat([qa, summ, data2txt])

            tmp["is_hal"] = tmp["labels"].apply(lambda x: len(x) != 0).astype("int")
            # add special tokens
            tmp["prompt"] = tmp["prompt"].apply(lambda x: f"<s>[INST] {x} [/INST]")
            tmp["response"] = tmp["response"].apply(lambda x: f"{x}</s>")

            tmp["response_len"] = tmp["response"].apply(lambda x: len(x.split(" ")))
        else:
            tmp = df[
                (df["task_type"] == self.task_type)
                & (df["model"] == model_n)
                & (df["quality"] == "good")
            ].copy()
            tmp["is_hal"] = tmp["labels"].apply(lambda x: len(x) != 0).astype("int")
            # add special tokens
            tmp["prompt"] = tmp["prompt"].apply(lambda x: f"<s>[INST] {x} [/INST]")
            tmp["response"] = tmp["response"].apply(lambda x: f"{x}</s>")

            tmp["response_len"] = tmp["response"].apply(lambda x: len(x.split(" ")))

            if self.task_type == "Data2txt":
                tmp["length"] = tmp["response"].apply(
                    lambda x: len(x.split(" "))
                ) + tmp["prompt"].apply(lambda x: len(x.split(" ")))

                cnt_grounded = len(tmp[tmp["is_hal"] == 0])
                keep_hallu = cnt_grounded * 2
                ids_to_keep = (
                    tmp[tmp["is_hal"] == 0]["id"].tolist()
                    + tmp[tmp["is_hal"] == 1]["id"].sample(keep_hallu).tolist()
                )
                tmp = tmp[tmp["id"].isin(ids_to_keep)]
                tmp = tmp[tmp["length"] <= 1024]
                tmp.drop(columns=["length"], inplace=True)

            if (self.task_type == "Summary") and (
                self.model_name == "Llama-2-13b-chat-hf"
            ):
                tmp["length"] = tmp["response"].apply(
                    lambda x: len(x.split(" "))
                ) + tmp["prompt"].apply(lambda x: len(x.split(" ")))
                tmp = tmp[tmp["length"] <= 1450]
                tmp.sort_values(by=["length"], ascending=False, inplace=True)
                tmp.drop(columns=["length"], inplace=True)

        return tmp

    def save_data(self, df: pd.DataFrame) -> None:
        """Saves the processed dataframe to a JSON file.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to save.

        """  # noqa: D401
        os.makedirs(f"{self.save_dir}/{self.model_name}/", exist_ok=True)
        df.to_json(f"{self.save_dir}/{self.model_name}/{self.task_type}.json")

    def load_cached_data(self) -> pd.DataFrame:
        """Loads cached data from the specified directory if it exists.

        Returns
        -------
        pd.DataFrame
            The loaded dataframe if the file exists, otherwise an empty dataframe.

        """  # noqa: D401
        file_path = f"{self.save_dir}/{self.model_name}/{self.task_type}.json"

        if os.path.exists(file_path):
            try:
                return pd.read_json(file_path)
            except ValueError as e:
                logger.info(f"Error loading cached data: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    def split_data(self, df: pd.DataFrame) -> tuple[np.ndarray[int], np.ndarray[int]]:
        """Splits the data into training and testing sets based on the specified method.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to split.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The training and testing dataframes.

        """  # noqa: D401
        indices = np.arange(len(df))  # Create an array of integer indices
        # self.val_size = len(indices)
        if self.val_size == len(indices):
            return None, indices
        train_test_indices, val_indices = train_test_split(
            indices, test_size=self.val_size, random_state=self.random_state
        )
        return train_test_indices, val_indices

    def process(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, tp.Optional[np.ndarray]]:
        """Execute the full data processing pipeline: loads presaved data if it exists,
        otherwise processes the raw data, filters, saves, and splits it.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The processed training and testing dataframes.

        """  # noqa: D205
        # Try to load presaved data
        logger.info("Trying to load cached data")
        tmp = self.load_cached_data()

        if not tmp.empty:
            logger.info("Loaded cached data successfully!")
        else:
            logger.info("Cache is empty, loading and processing raw data...")

            # If presaved data does not exist, load raw data
            responses_df, source_df = self.load_data()
            if responses_df.empty or source_df.empty:
                raise ValueError(
                    "Error: One or both of the loaded dataframes are empty. Cannot proceed with processing."
                )

            # Filter and label the data
            tmp = self.filter_and_label_data(source_df, responses_df)

            # Save the processed data
            self.save_data(tmp)
            logger.info("Processed data is saved successfully!")

        # add dataset name
        tmp["name"] = "ragtruth"
        # Split the data into train and test sets
        train_indices, test_indices = self.split_data(tmp)
        X, y = tmp.drop(columns=["is_hal"]), tmp["is_hal"]

        return X, y, train_indices, test_indices
