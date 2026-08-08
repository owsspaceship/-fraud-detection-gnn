"""
Feature engineering for the fraud detection tabular and graph models.
"""

import pandas as pd
import networkx as nx


def add_balance_error_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add balance-reconciliation discrepancy features.

    `errorBalanceOrig` in particular is a very strong signal in PaySim
    (78% of XGBoost feature importance in notebook 06) but is very likely
    a simulator artefact -- the fraud-generation logic updates the origin
    balance more cleanly than legitimate transaction logic does. See the
    README "Limitations" section before relying on this in a real-world
    setting.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction dataframe with oldbalanceOrg, newbalanceOrig, amount,
        oldbalanceDest, newbalanceDest columns.

    Returns
    -------
    pd.DataFrame
        Same dataframe with errorBalanceOrig and errorBalanceDest added.
    """
    df = df.copy()
    df['errorBalanceOrig'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    return df


def add_type_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the transaction type (TRANSFER / CASH_OUT only)."""
    df = df.copy()
    df['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)
    df['type_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype(int)
    return df


def add_graph_degree_features(df: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    """
    Attach sender/receiver in-degree and out-degree to each transaction row.

    Tested in notebook 07 against an honest tabular baseline: did not
    improve F1, precision, or PR-AUC. Included here for completeness and
    reproducibility of that comparison, not because it's recommended for
    production use on this dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction dataframe with nameOrig, nameDest columns.
    G : nx.DiGraph
        The transaction graph (see graph_construction.build_transaction_graph).

    Returns
    -------
    pd.DataFrame
        Same dataframe with four degree columns added.
    """
    df = df.copy()
    in_degree_dict = dict(G.in_degree())
    out_degree_dict = dict(G.out_degree())

    df['orig_out_degree'] = df['nameOrig'].map(out_degree_dict)
    df['orig_in_degree'] = df['nameOrig'].map(in_degree_dict)
    df['dest_in_degree'] = df['nameDest'].map(in_degree_dict)
    df['dest_out_degree'] = df['nameDest'].map(out_degree_dict)

    return df


# Feature set definitions, matching the three models compared in notebooks 06-07
NAIVE_BASELINE_FEATURES = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'errorBalanceOrig', 'errorBalanceDest',
    'type_TRANSFER', 'type_CASH_OUT'
]

HONEST_BASELINE_FEATURES = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'type_TRANSFER', 'type_CASH_OUT'
]

GRAPH_ENHANCED_FEATURES = HONEST_BASELINE_FEATURES + [
    'orig_out_degree', 'orig_in_degree',
    'dest_in_degree', 'dest_out_degree'
]