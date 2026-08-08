"""
XGBoost model training for fraud classification.
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


def make_train_test_split(X: pd.DataFrame, y: pd.Series,
                            test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train/test split, preserving the fraud class ratio in both
    sets. Essential given the severe class imbalance (~0.3% fraud rate) --
    an unstratified split risks a test set with almost no fraud examples.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def train_fraud_classifier(X_train: pd.DataFrame, y_train: pd.Series,
                             n_estimators: int = 100,
                             max_depth: int = 6,
                             learning_rate: float = 0.1,
                             random_state: int = 42) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier with class-imbalance handling via
    scale_pos_weight, tuned automatically from the training label ratio.

    Parameters
    ----------
    X_train, y_train : training features and labels
    n_estimators, max_depth, learning_rate : XGBoost hyperparameters
        (baseline defaults, not extensively tuned -- see README)
    random_state : for reproducibility

    Returns
    -------
    xgb.XGBClassifier
        The fitted model.
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric='aucpr'
    )
    model.fit(X_train, y_train)
    return model