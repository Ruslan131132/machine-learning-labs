from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DATASET_PATH = DATA_DIR / "new_data.csv"

TARGET = "attack"

# 1. Train and evaluate the classifier on the dataset preprocessed in lab 3, then plot the tree
data_frame = pd.read_csv(DATASET_PATH, index_col=0)
X = data_frame.drop(TARGET, axis=1)
Y = data_frame[TARGET]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

dtree = DecisionTreeClassifier(criterion="entropy", max_depth=2, min_samples_leaf=37)
dtree.fit(X_train, Y_train)

OUTPUT_DIR.mkdir(exist_ok=True)
plt.figure()
tree.plot_tree(dtree, feature_names=list(X.columns), class_names=["0", "1"], filled=True)
plt.savefig(OUTPUT_DIR / "decision-tree.pdf")
plt.show()

Y_prediction = dtree.predict(X_test)
print("Classifier accuracy:", accuracy_score(Y_test, Y_prediction))
