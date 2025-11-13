import typing as tp
from ..methods.hallucination_detection_abc import HallucinationDetectionMethod, T
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc

def get_results(
    model: HallucinationDetectionMethod,
    X: tp.Sequence[T],
    y: list[int],
) -> pd.DataFrame:
    
    y_pred_score = model.predict_score(X)
    y_pred = model.predict(X)

    # Calculate metrics
    roc_auc = roc_auc_score(y, y_pred_score)
    f1 = f1_score(y, y_pred)

    # Generate precision-recall curve
    precision, recall, _ = precision_recall_curve(y, y_pred_score)

    metrics_current = {
        "dataset": 'test',
        "roc_auc": roc_auc,
        "f1": f1,
        "pr_auc": auc(recall, precision),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "split" : 0
    }

    return [metrics_current]




