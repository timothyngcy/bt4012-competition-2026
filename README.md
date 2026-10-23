# BT4012 Competition 2026 — Fraudulent Bitcoin Transaction Detection

## Problem

Each Bitcoin transaction is a node in a graph, and each edge is a flow of funds
from one transaction to another. Some transactions are labelled licit or illicit;
most are unlabelled. The task is to predict the probability that each test
transaction is illicit, scored by ROC-AUC.

Training covers time steps 1–35 and the test set covers steps 36–49, so the model
has to generalise to a later period it has never seen.

## Data

Place the competition files in `data/`:

| File | Contents |
|------|----------|
| `train.csv` | Transactions from steps 1–35 with 165 anonymised features and a label (1 = illicit, 0 = licit, blank = unknown) |
| `test.csv` | Transactions from steps 36–49 to predict |
| `txs_edgelist.csv` | Directed edges between transactions |
| `sample_submission.csv` | Submission format |

## Notebook

[`e1122822.ipynb`](e1122822.ipynb) contains the full pipeline:

1. Data audit and exploratory analysis
2. Validation that trains on earlier time steps and predicts the next 14, matching the test gap
3. Feature engineering: correlation pruning plus simple graph features
4. Models: gradient boosting (HGB), two neural networks and a LightGBM that also learns from unlabelled rows
5. A rank-weighted blend of the four models, refined with graph and similarity adjustments and self-training
6. Final training on all labelled data and writing the submission

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Select `.venv` as the notebook kernel and run all cells from top to bottom.
The submission is written to `submissions/submission_final.csv`.
