# Lab 7 — Support Vector Machine classifier

## Task

Using the dataset from [lab 3](../lab03-data-preprocessing), build a classifier based on the support vector machine:

1. Create, train and evaluate the classifier.
2. Reject objects whose features fall outside the admissible ranges.
3. Demonstrate the classifier on a new object with decision probabilities.

## Approach

- 75/25 train/test split with a fixed `random_state`.
- `SVC(probability=True)` with the default RBF kernel; `probability=True` enables `predict_proba`
  through Platt scaling.
- The anomaly check is the same z-score rule as in [lab 4](../lab04-knn-classifier).

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | **0.813** |
| Sample object | not anomalous |
| Prediction for the sample object | class `1` (attack), probabilities `[0.394, 0.606]` |

With default hyperparameters (`C=1.0`, `gamma="scale"`) the SVM is on par with Naive Bayes and noticeably less
confident than the tree-based and distance-based models. Most flows are packed close to zero after standardization,
so the RBF kernel needs tuning of `C` and `gamma` to separate the classes well.

The script also contains the PCA and cross-validation experiments for this model — see [lab 8](../lab08-pca-and-cross-validation) for the comparison.

## Run

```bash
# data/new_data.csv is required (output of lab 3)
python3 lab07-svm/svm_classifier.py
```
