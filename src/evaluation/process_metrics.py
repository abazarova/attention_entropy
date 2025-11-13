from typing import Optional

import numpy as np
import pandas as pd
from comet_ml import Experiment
from loguru import logger
from scipy.interpolate import interp1d
from tabulate import tabulate


def interpolate_precision_recall(curves, mean_recall):
    """Interpolate and average precision-recall curves."""
    interpolated_precisions = [
        interp1d(recall, precision)(mean_recall) for precision, recall in curves
    ]
    return np.mean(interpolated_precisions, axis=0)


def log_metrics(df, experiment, dataset):
    """Log dataset-specific metrics."""
    df_dataset = df[df["dataset"] == dataset]
    records = df_dataset.drop(columns="dataset").to_dict("records")
    for record in records:
        metrics_to_log = {
            f"{dataset}_{k}": v for k, v in record.items() if k != "seed"
        }
        experiment.log_metrics(metrics_to_log, step=record["seed"])


def log_pr_curves(df, experiment, dataset):
    """Log and collect precision-recall curves."""
    df_dataset = df[df["dataset"] == dataset]

    dataset_curves = []
    for _, row in df_dataset.iterrows():
        experiment.log_curve(
            f"{dataset}_precision_recall_curve",
            x=row["recall"],
            y=row["precision"],
            step=row["seed"],
        )
        dataset_curves.append((row["precision"], row["recall"]))
    return dataset_curves


def process_metrics(metrics: list[dict[str, str | float]], experiment: Optional[Experiment] = None):
    if experiment is not None:
        experiment.log_asset_data(metrics, name="metrics.json")

    df = pd.DataFrame(metrics)

    # TODO: fixthis
    # This code is bad! But consider it as temporary solution
    if len(metrics) == 1:
        # Aggregation and formatting of the metrics
        agg_df = df.groupby("dataset")[["roc_auc", "pr_auc", "f1"]].agg(["mean", "std"])
        formatted_df = agg_df.apply(
            lambda row: row.xs("mean", level=1).apply(lambda x: f"{x:.3f}")
            + " ± "
            + row.xs("std", level=1).apply(lambda x: f"{x:.3f}"),
            axis=1,
        ).loc[["test"], :]
        if experiment is not None:
            experiment.log_table("Aggregated Metrics Table (Raw).csv", agg_df)
            experiment.log_table("Aggregated Metrics Table.csv", formatted_df)

        return tabulate(
            formatted_df, headers="keys", tablefmt="pipe", showindex=True
        ), agg_df

    # Mean recall for interpolation
    mean_recall = np.linspace(0, 1, 100)

    for dataset in df["dataset"].unique():
        if experiment is not None:
            log_metrics(df.drop(columns=["precision", "recall"]), experiment, dataset)

        # Log PR curves and compute the mean PR curve
        dataset_curves = log_pr_curves(df, experiment, dataset)
        mean_precision = interpolate_precision_recall(dataset_curves, mean_recall)
        if experiment is not None:
            experiment.log_curve(
                f"mean_{dataset}_precision_recall_curve", x=mean_recall, y=mean_precision
            )

    # Aggregation and formatting of the metrics
    agg_df = df.groupby("dataset")[["roc_auc", "pr_auc", "f1"]].agg(["mean", "std"])

    formatted_df = agg_df.apply(
        lambda row: row.xs("mean", level=1).apply(lambda x: f"{x:.3f}")
        + " ± "
        + row.xs("std", level=1).apply(lambda x: f"{x:.3f}"),
        axis=1,
    ).loc[["train", "val", "test"], :]

    if experiment is not None:
        experiment.log_table("Aggregated Metrics Table (Raw).csv", agg_df)
        experiment.log_table("Aggregated Metrics Table.csv", formatted_df)

    return tabulate(
        formatted_df, headers="keys", tablefmt="pipe", showindex=True
    ), agg_df
