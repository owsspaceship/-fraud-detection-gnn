# Fraud Detection in Finacial Transaction Networks

## Overview
A graph-based machine learning system for detecting fradualnt trasctions in finacial networks. Built using the PaySim synthetic transaction dataset, thisproject models accounts as nodes and transactions asweighted directed adges then applies spectral graph theory and machine learning tp identify fradulent behaviour.

## Motivation
Fraud detection is a core problem is a core problem in fintech and credit scoring. Traditional ML approach treat transactions independently - this project captures the  ** network strucure** of transactions, which is where fraud patterns oftern hide.

## Approach
1. **Graph Construction** — model the transaction network as a directed weighted graph
2. **Spectral Analysis** — compute graph Laplacian and extract eigenvalue-based features
3. **Node Embeddings** — learn low-dimensional account representations
4. **Classification** — train a model to predict fraudulent accounts/transactions
5. **Evaluation** — benchmark against traditional baselines

## Tech Stack
- Python, NetworkX, NumPy, Pandas
- Scikit-learn, PyTorch
- Jupyter Notebooks

## Project Structure
fraud-detection-gnn/
├── data/ # raw and processed data
├── notebooks/# exploration and analysis
├── src/# core source code
├── results/# figures and outputs
└── requirements.txt

## Status
In progress — graph construction phase

## Author
Owen — Mathematics & Statistics and Data Science, University of Cape Town