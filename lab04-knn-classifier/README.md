# Lab 4 — k-Nearest Neighbors classifier

## Task

Using the dataset from [lab 3](../lab03-data-preprocessing), build a classifier based on the k-nearest neighbors method:

1. Create, train and evaluate the classifier.
2. Implement a check that the features of a classified object lie within the admissible ranges
   and reject anomalous objects before they reach the classifier.
3. Demonstrate the classifier on new objects, printing the decision probabilities.
   Input objects are preprocessed with the same scheme as the training data.

## Approach

- 75/25 train/test split with a fixed `random_state`.
- `KNeighborsClassifier(n_neighbors=3, metric="euclidean")`.
- Features are standardized in lab 3, so an object is treated as anomalous when any feature has `|z| > 3` —
  exactly the rule used to remove outliers from the training data.
- The script asks for a comma-separated object; an empty input falls back to a sample flow from the dataset.

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | **0.956** |
| Sample object | not anomalous |
| Prediction for the sample object | class `1` (attack), probabilities `[0.0, 1.0]` |

## Run

```bash
# data/new_data.csv is required (output of lab 3)
python3 lab04-knn-classifier/knn_classifier.py
```
