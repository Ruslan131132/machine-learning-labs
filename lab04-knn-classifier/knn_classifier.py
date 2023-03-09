from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_PATH = DATA_DIR / "new_data.csv"

TARGET = "attack"

# 1. Train and evaluate the classifier on the dataset preprocessed in lab 3
data_frame = pd.read_csv(DATASET_PATH, index_col=0)
X = data_frame.drop(TARGET, axis=1)
Y = data_frame[TARGET]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn.fit(X_train, Y_train)
Y_prediction = knn.predict(X_test)
print("Classifier accuracy:", accuracy_score(Y_test, Y_prediction))
