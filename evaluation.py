import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(model, X, y):
    """Evaluate a fitted binary classifier, with malignant encoded as 1."""
    predictions = model.predict(X)
    positive_column = list(model.classes_).index(1)
    scores = model.predict_proba(X)[:, positive_column]

    # ROC AUC requires both target classes in the uploaded data.
    auc = (
        roc_auc_score(y, scores)
        if len(np.unique(y)) == 2
        else np.nan
    )

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": auc,
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }
    return metrics, predictions