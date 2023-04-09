# Lab 6 — Naive Bayes classifier

## Task

Using the dataset from [lab 3](../lab03-data-preprocessing), build a classifier based on the naive Bayes method:

1. Create, train and evaluate the classifier.
2. Reject objects whose features fall outside the admissible ranges.
3. Demonstrate the classifier on a new object with decision probabilities.

## Approach

- 75/25 train/test split with a fixed `random_state`.
- `GaussianNB` — every feature is modeled with a normal distribution per class, features are assumed independent.
- The anomaly check is the same z-score rule as in [lab 4](../lab04-knn-classifier).

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | **0.810** |
| Sample object | not anomalous |
| Prediction for the sample object | class `1` (attack), probabilities `[0.00006, 0.99994]` |

Naive Bayes is noticeably weaker than k-NN and the decision tree on this data: traffic features such as
`bytes`/`sbytes` or `rate`/`srate` are strongly correlated, which breaks the independence assumption,
and their distributions are far from normal even after standardization.

## Run

```bash
# data/new_data.csv is required (output of lab 3)
python3 lab06-naive-bayes/naive_bayes.py
```
