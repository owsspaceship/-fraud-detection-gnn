# Fraud Detection in Financial Transaction Networks: Does Graph Structure Add Value?

## Overview

A rigorous empirical investigation into whether graph-based machine learning 
methods improve fraud detection over strong tabular baselines, using the 
PaySim synthetic financial transaction dataset (6.3M transactions). This 
project models the transaction network as a directed graph, tests spectral 
graph theory and embedding methods against that graph, and benchmarks all 
graph-derived features against an honest tabular baseline.

**The central finding: across PageRank, Node2Vec embeddings, and explicit 
degree features, no graph technique improved fraud detection over a 
well-engineered tabular baseline on this dataset — a result reached through 
systematic testing, not assumption, with the specific structural reasons 
why identified and explained at every step.**

This project prioritises methodological honesty over impressive-sounding 
results. Every technique was tested against evidence; several were 
deliberately dropped when the evidence didn't support keeping them.

## Motivation

Fraud detection is a core problem in fintech and credit risk. The intuitive 
appeal of graph methods is that fraud often involves networks of related 
accounts, not isolated transactions. This project set out to test that 
intuition rigorously on a large (2.77M transaction) dataset rather than 
assume it — and to be honest about what the evidence actually showed, 
including when it didn't support the graph-based hypothesis.

## Key Findings

### 1. Fraud has a clear structural signature (Notebook 01)
Fraud originators (in-degree 0, out-degree 1) and fraud receivers (in-degree 
5.66 vs 0.84 for normal accounts) show sharply different degree patterns — 
a genuine structural signal, established through exploratory analysis.

### 2. Fraud networks show temporal regime change (Notebook 03)
The dataset splits into two structurally distinct regimes: a stable 
low-fraud-density period (0.15% fraud rate) and a later high-density period 
(2.28% fraud rate, 15x higher) with completely disjoint fraud actors (zero 
shared originators between regimes). Fraud accounts — senders and receivers 
alike — are used exactly once and never reused, in both regimes.

### 3. Spectral graph theory methods add no discriminative signal (Notebook 04)
PageRank (full graph and fraud subgraph) correlates perfectly (r=1.0000) 
with in-degree — fully redundant. Laplacian connected-components theorem 
verified empirically (99.6% match via sparse eigensolver). Component size 
and spectral gap are statistically identical between fraud and normal 
subgraphs. Root cause: this network is dominated by tiny, mostly one-off 
account interactions (88% of nodes have degree 1) with essentially no 
multi-hop connectivity for spectral methods to exploit.

### 4. Node2Vec is structurally infeasible on this graph (Notebook 05)
A pre-training diagnostic found the largest connected component in the 
*entire 3.27M-node graph* is only 90 nodes. Random-walk-based embeddings 
require much longer walks than this graph's topology permits anywhere. 
Confirmed via a fast structural check before committing to a training run, 
avoiding wasted compute on a result that PageRank's failure had already 
foreshadowed.

### 5. A single engineered feature dominates the naive baseline (Notebook 06)
`errorBalanceOrig` — a balance-reconciliation discrepancy — carries 78% of 
feature importance in an XGBoost baseline achieving F1=0.945. This is very 
likely a PaySim simulator artefact (the fraud-generation logic updates 
balances more "cleanly" than legitimate transaction logic), not a 
real-world fraud pattern. Removing it drops F1 to 0.438 — precision falls 
from 89.9% to 28.1%, while recall stays high (~99.5%).

### 6. Graph features do not close the precision gap (Notebook 07)
Adding in-degree/out-degree to the honest baseline (excluding the simulator 
artefact) did not improve performance — F1 fell slightly (0.438 → 0.420). 
This is consistent with findings 3 and 4: this network simply lacks the 
structure graph methods are built to exploit, and XGBoost's tree splits 
likely already approximate degree-equivalent signal through amount and 
balance correlations.

## Methodology

Every major methodological decision in this project was pressure-tested 
through a structured five-perspective critique (Contrarian, Assumption-Ripper, 
Expansionist, Outsider, and a synthesising Chairman) before being acted on — 
including the decision to defer PageRank computation, the decision to test 
rather than assume Node2Vec's viability, and the decision to build an honest 
(artefact-excluded) baseline rather than rely on the inflated naive one.

## Tech Stack
- Python, NetworkX, NumPy, Pandas, SciPy
- XGBoost, scikit-learn
- Jupyter Notebooks, Matplotlib

## Project Structure
fraud-detection-gnn/
├── data/ # raw and processed data (not tracked in git)
├── notebooks/
│ ├── 01_exploration.ipynb # EDA, class imbalance, degree signatures
│ ├── 02_graph_construction.ipynb # clean graph build, node/edge attributes
│ ├── 03_temporal_analysis.ipynb # cycle detection, regime discovery
│ ├── 04_spectral_analysis.ipynb # PageRank, Laplacian, spectral gap testing
│ ├── 05_embeddings.ipynb # Node2Vec feasibility diagnostic
│ ├── 06_baseline_model.ipynb # XGBoost tabular baseline (naive + honest)
│ └── 07_graph_model_and_evaluation.ipynb # graph features vs honest baseline
├── results/figures/ # saved visualisations
├── src/ # reusable source modules
└── requirements.txt

## Setup
1. Clone the repo
2. Create and activate virtual environment
3. Install dependencies
4. Download the data
5. Run notebooks in order, 01 through 07

## Limitations

- **PaySim is synthetic data.** Several findings — most notably 
  `errorBalanceOrig`'s dominance — are very likely artefacts of the 
  simulator's fraud-generation logic rather than patterns that would hold 
  in real financial data. This is stated explicitly rather than glossed 
  over, and is the main reason this project's conclusions about graph 
  methods should not be over-generalised to real-world fraud detection.
- **This network's sparsity is likely dataset-specific.** A real-world 
  network with repeat customers (e.g. merchant/consumer transactions) would 
  have different — likely richer — connectivity, where graph and embedding 
  methods may perform very differently. This is a property of PaySim's P2P 
  transaction structure, not a general statement about graph methods in 
  fraud detection.

## Future Work

- **Regime-aware modelling**: formal Markov-switching / Hidden Markov Model 
  treatment of the two temporal regimes identified in notebook 03, once 
  covered in ongoing coursework (MIT OCW Stochastic Processes)
- **Cross-dataset validation**: testing the tabular baseline's engineered 
  features (excluding PaySim-specific artefacts) against a real-world 
  dataset (e.g. IEEE-CIS) to test generalisability
- **Temporal cross-cycle pattern matching**: testing whether structurally 
  similar fraud patterns recur across time cycles even when account 
  identities never repeat, requiring aligned embedding spaces across 
  separately-trained cycles
- **Neighbourhood-aggregation embeddings** (e.g. single-hop GraphSAGE-style 
  methods) as a walk-free alternative better suited to this graph's sparsity

## Author
Owen — Mathematics & Mathematical Statistics, University of Cape Town