import os
import typing as tp

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from ..methods.hallucination_detection_abc import HallucinationDetectionMethod


def evaluate(
    model: HallucinationDetectionMethod,
    X: pd.DataFrame,
    y: tp.Union[pd.Series, list[int]],
    k: int = 5,
    seed: int = 42,
    test_size: float = 0.25,
    val_size: tp.Union[float, int] = 0.15,
    tune_hyperparameters: bool = True,
    pretrained: bool = False,
    save_best_model: bool = True,
    model_save_path: str = "./best_model.pkl",
) -> list[dict[str, str | float]]:
    """Evaluate the performance of a given model with pandas DataFrames.

    Args:
        model (HallucinationDetectionMethod): The model to be evaluated, with `fit`, `predict`,
            `predict_score`, and `transform` methods.
        X (pd.DataFrame): All features as a DataFrame.
        y (pd.Series or list): All labels.
        k (int): The number of splits for cross-validation.
        seed (int): Random seed for reproducibility.
        test_size (float): Proportion of data to use for test set.
        val_size (float or int): Proportion of training data to use for validation set (if float)
            or absolute number of validation samples (if int).
        tune_hyperparameters (bool): Whether to select method hyperparameters using val set
            (or use the ones from the config).
        pretrained (bool): Whether the model is already trained.
        save_best_model (bool): Whether to save the best model based on test performance.
        model_save_path (str): Path to save the best model.

    Returns:
        list: A list of dictionaries containing performance metrics for each dataset and split.
    """
    result = []
    best_model = None
    best_score = -1
    best_seed = -1

    # Convert y to pandas Series if it's a list
    if isinstance(y, list):
        y = pd.Series(y)

    idxs = np.arange(len(X))

    for random_state in range(seed, seed + k):
        train_val_idxs, test_idxs = train_test_split(
            idxs, test_size=test_size, random_state=random_state
        )

        # Handle both float and integer val_size
        if isinstance(val_size, float):
            val_numel = int(len(train_val_idxs) * val_size)
        elif isinstance(val_size, int):
            val_numel = val_size
            # Ensure val_size doesn't exceed available training samples
            if val_numel > len(train_val_idxs):
                raise ValueError(
                    f"val_size ({val_numel}) cannot be larger than the number of training samples ({len(train_val_idxs)})"
                )
        else:
            raise TypeError(
                "val_size must be either float (proportion) or int (absolute number)"
            )

        train_idxs, val_idxs = train_val_idxs[:-val_numel], train_val_idxs[-val_numel:]

        if tune_hyperparameters:
            model.reset()
            X_val, y_val = X.iloc[val_idxs], y.iloc[val_idxs]
            model.fit_hyperparameters(X_val, y_val)

        X_transformed = model.transform(X.copy())
        # ### DELETE AFTERWARDS
        # lengths = X["response"].apply(lambda x: len(x.split(" ")))
        # X_transformed = [
        #     elem for i, elem in enumerate(X_transformed) if lengths[i] == 1
        # ]
        # y_ = y.values
        # y_ = [elem for i, elem in enumerate(y_) if lengths[i] == 1]
        # print(np.unique(y_, return_counts=True))
        # print(f"{len(y_)} SAMPLES LEFT")
        # idxs = np.arange(len(X_transformed))
        # train_val_idxs, test_idxs = train_test_split(
        #     idxs, test_size=test_size, random_state=random_state, stratify=y_
        # )

        # train_idxs, val_idxs = train_test_split(
        #     train_val_idxs,
        #     test_size=val_numel,
        #     random_state=random_state,
        #     stratify=[y_[i] for i in train_val_idxs],
        # )
        # # train_val_idxs[:-val_numel], train_val_idxs[-val_numel:]
        # y_train, y_val, y_test = (
        #     [y_[i] for i in train_idxs],
        #     [y_[i] for i in val_idxs],
        #     [y_[i] for i in test_idxs],
        # )
        # ### END OF DELETE SECTION

        X_train = [X_transformed[i] for i in train_idxs]
        X_val = [X_transformed[i] for i in val_idxs]
        X_test = [X_transformed[i] for i in test_idxs]

        y_train, y_val, y_test = (
            y.iloc[train_idxs].values.tolist(),
            y.iloc[val_idxs].values.tolist(),
            y.iloc[test_idxs].values.tolist(),
        )

        if not pretrained:
            model.fit(X_train, y_train, X_val, y_val)
            model.fit_threshold(X_val, y_val)

        # Store current model for potential best model selection
        current_model = model.clone() if hasattr(model, "clone") else model

        # Evaluate on train, val, and test sets
        test_roc_auc = None
        for dataset, X_split, y_split in zip(
            ["train", "val", "test"], [X_train, X_val, X_test], [y_train, y_val, y_test]
        ):
            # Predict probabilities and classes
            y_pred_score = model.predict_score(X_split)
            y_pred = model.predict(X_split)
            # Calculate metrics
            try:
                roc_auc = roc_auc_score(y_split, y_pred_score)
            except:
                breakpoint()
            f1 = f1_score(y_split, y_pred)

            # Generate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_split, y_pred_score)

            metrics_current = {
                "dataset": dataset,
                "roc_auc": roc_auc,
                "f1": f1,
                "pr_auc": auc(recall, precision),
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "seed": random_state,
            }

            result.append(metrics_current)

            # Store test ROC AUC for best model selection
            if dataset == "test":
                test_roc_auc = roc_auc

        # Update best model if current test performance is better
        if test_roc_auc is not None and test_roc_auc > best_score:
            best_score = test_roc_auc
            best_seed = random_state
            best_model = current_model

    # Save the best model if requested
    if save_best_model and best_model is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(best_model, model_save_path)
        print(
            f"Best model saved with test ROC AUC: {best_score:.4f} (seed: {best_seed})"
        )

    return result, best_model
