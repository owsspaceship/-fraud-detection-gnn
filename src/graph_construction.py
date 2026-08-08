"""
Graph construction utilities for the fraud detection transaction network.

Builds a directed, weighted graph from PaySim transaction data, with node
attributes (in-degree, out-degree, fraud labels) attached for downstream use.
"""

import pandas as pd
import networkx as nx
import pickle


def filter_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the standard project filters to raw PaySim transaction data.

    Fraud in PaySim occurs exclusively in TRANSFER and CASH_OUT transactions
    (established in notebook 01). Zero-amount transactions are removed as a
    simulator data quality artefact.

    Parameters
    ----------
    df : pd.DataFrame
        Raw PaySim transaction dataframe.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe, index reset.
    """
    df_filtered = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]
    df_filtered = df_filtered[df_filtered['amount'] > 0]
    return df_filtered.reset_index(drop=True)


def build_transaction_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed weighted graph from filtered transaction data.

    Nodes are accounts (nameOrig, nameDest). Edges are transactions, directed
    from sender to receiver, weighted by transaction amount. Edge attributes
    also include transaction type, fraud label, and time step.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered transaction dataframe (see filter_transactions).

    Returns
    -------
    nx.DiGraph
        The constructed transaction graph, without node attributes attached
        yet (see attach_node_attributes).
    """
    G = nx.DiGraph()

    edges = list(zip(
        df['nameOrig'], df['nameDest'], df['amount'],
        df['type'], df['isFraud'], df['step']
    ))

    for orig, dest, amount, txn_type, is_fraud, step in edges:
        G.add_edge(orig, dest,
                    weight=amount,
                    txn_type=txn_type,
                    isFraud=is_fraud,
                    step=step)

    return G


def attach_node_attributes(G: nx.DiGraph, df: pd.DataFrame) -> nx.DiGraph:
    """
    Attach in-degree, out-degree, and fraud role labels to every node.

    Note: PageRank is deliberately NOT computed here. It was tested in
    notebook 04 (full graph and fraud subgraph) and found to correlate
    perfectly (r=1.0000) with in-degree due to this network's sparse,
    largely disconnected structure -- see notebook 04 for the full
    justification. Adding it here would be redundant.

    Parameters
    ----------
    G : nx.DiGraph
        Graph built by build_transaction_graph.
    df : pd.DataFrame
        The same filtered dataframe used to build G, used to identify
        fraud originator/receiver accounts.

    Returns
    -------
    nx.DiGraph
        The same graph object, with node attributes set in place.
    """
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    fraud_originators = set(df[df['isFraud'] == 1]['nameOrig'])
    fraud_receivers = set(df[df['isFraud'] == 1]['nameDest'])

    for node in G.nodes():
        G.nodes[node]['in_degree'] = in_deg[node]
        G.nodes[node]['out_degree'] = out_deg[node]
        G.nodes[node]['is_fraud_originator'] = 1 if node in fraud_originators else 0
        G.nodes[node]['is_fraud_receiver'] = 1 if node in fraud_receivers else 0

    return G


def save_graph(G: nx.DiGraph, path: str) -> None:
    """Serialise a graph to disk using pickle's highest protocol."""
    with open(path, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(path: str) -> nx.DiGraph:
    """Load a previously saved graph from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_and_save_transaction_graph(raw_csv_path: str,
                                       filtered_csv_out: str,
                                       graph_pkl_out: str) -> nx.DiGraph:
    """
    Full pipeline: load raw data, filter, build graph, attach attributes,
    save both the filtered dataframe and the graph to disk.

    This is the function notebook 02 effectively runs cell-by-cell --
    provided here as a single reusable entry point.
    """
    df = pd.read_csv(raw_csv_path)
    df_filtered = filter_transactions(df)
    df_filtered.to_csv(filtered_csv_out, index=False)

    G = build_transaction_graph(df_filtered)
    G = attach_node_attributes(G, df_filtered)
    save_graph(G, graph_pkl_out)

    return G