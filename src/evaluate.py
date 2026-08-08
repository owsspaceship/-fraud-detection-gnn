"""
Evaluation utilities for the fraud detection models.

Accuracy is deliberately not used as a primary metric -- with a ~0.3% fraud
rate, a model predicting "not fraud" for everything scores ~99.7% accuracy
while catching zero fraud. See notebook 01 for the full discussion.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                               average_precision_score, classification_report,
                               confusion_matrix)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                    label: str = "") -> dict:
    """
    Compute the full evaluation suite for a trained fraud classifier.

    Parameters
    ----------
    model : fitted classifier with .predict and .predict_proba
    X_test, y_test : held-out test features and labels
    label : optional name for this model run, printed with results

    Returns
    -------
    dict
        precision, recall, f1, pr_auc, and the confusion matrix.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    results = {
        'label': label,
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'pr_auc': average_precision_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }

    if label:
        print(f"=== {label} ===")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"PR-AUC:    {results['pr_auc']:.4f}")

    return results


def compare_models(*results: dict) -> pd.DataFrame:
    """
    Build a side-by-side comparison table from multiple evaluate_model()
    outputs -- mirrors the comparison table used in notebook 07 to test
    whether graph features improve on the honest tabular baseline.
    """
    rows = []
    for r in results:
        rows.append({
            'model': r['label'],
            'precision': r['precision'],
            'recall': r['recall'],
            'f1': r['f1'],
            'pr_auc': r['pr_auc'],
        })
    return pd.DataFrame(rows)