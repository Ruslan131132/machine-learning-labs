# Lab 8 — PCA projection and cross-validation

## Task

For every classifier from labs 4–7:

1. Repeat the experiment after projecting the data into the space of principal components and compare the results.
2. Evaluate the model with cross-validation and compare the results.

## Implementation

The experiments extend the classifier scripts directly, so each model is compared against itself on the same split:

| Classifier | Script | PCA components | CV folds |
|------------|--------|----------------|----------|
| k-NN | [`knn_classifier.py`](../lab04-knn-classifier/knn_classifier.py) | 2 | 6 |
| Decision tree | [`decision_tree.py`](../lab05-decision-tree/decision_tree.py) | 15 (all) | 5 |
| Gaussian Naive Bayes | [`naive_bayes.py`](../lab06-naive-bayes/naive_bayes.py) | 15 (all) | 5 |
| SVM | [`svm_classifier.py`](../lab07-svm/svm_classifier.py) | 15 (all) | 5 |

- PCA is fitted with `sklearn.decomposition.PCA` and the same 75/25 split is used as in the base experiment.
- Cross-validation uses `KFold` + `cross_val_score` on the original feature space.

## Results

| Classifier | Base accuracy | Accuracy with PCA | Mean CV accuracy | CV scores per fold |
|------------|---------------|-------------------|------------------|--------------------|
| k-NN | 0.956 | 0.939 | 0.946 | 0.964, 0.959, 0.964, 0.921, 0.972, 0.895 |
| Decision tree | **0.977** | 0.932 | **0.966** | 0.973, 0.955, 0.970, 0.980, 0.952 |
| Naive Bayes | 0.810 | 0.859 | 0.890 | 0.986, 0.986, 0.702, 0.782, 0.995 |
| SVM | 0.813 | 0.812 | 0.682 | 0.692, 0.697, 0.736, 0.811, 0.475 |

## Conclusions

- **The decision tree is the best model** on this dataset both on the hold-out split and in cross-validation,
  and its scores are stable across folds.
- **k-NN loses only ~1.7%** after compressing 15 features into 2 principal components — the class structure
  is mostly captured by the first two components (see [lab 9](../lab09-pca-visualization)).
- **The decision tree gets worse with PCA**: rotated components mix the original features, so single-feature
  thresholds such as `sbytes <= -0.012` are no longer available.
- **Naive Bayes improves with PCA** (0.810 → 0.859): principal components are uncorrelated,
  which is closer to the independence assumption of the model.
- **SVM is unstable in cross-validation.** `KFold` is used without shuffling, and the balanced dataset is stored
  as all attacks followed by all normal flows, so some folds are dominated by one class. Models that are sensitive
  to the class mix (SVM, Naive Bayes) show a large spread between folds; `KFold(shuffle=True)` or `StratifiedKFold`
  would give a fairer estimate.
