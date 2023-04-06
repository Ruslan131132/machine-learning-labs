import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_PATH = DATA_DIR / "new_data.csv"

TARGET = "attack"
Z_SCORE_LIMIT = 3
# A standardized flow from the processed dataset
SAMPLE_OBJECT = (
    "-0.012376841893742508,-0.010681547862625913,-0.02659963953603794,-0.11484877128914105,"
    "-0.04101868780921577,-0.015478520643261943,-0.10818990822070183,-0.10915669863598569,"
    "-0.013416142300158071,-0.007896574517276864,-0.011544677753023085,-0.0074515333031985025,"
    "-0.19532659519987733,-0.07927406549426327,-0.1011929196142962"
)


def is_anomaly(objects):
    """Return True if any feature value lies outside the admissible z-score range."""
    return bool(np.any(np.abs(objects) > Z_SCORE_LIMIT))


# 1. Train and evaluate the classifier on the dataset preprocessed in lab 3
data_frame = pd.read_csv(DATASET_PATH, index_col=0)
X = data_frame.drop(TARGET, axis=1)
Y = data_frame[TARGET]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

gnb = GaussianNB()
gnb.fit(X_train, Y_train)
Y_prediction = gnb.predict(X_test)
print("Classifier accuracy:", accuracy_score(Y_test, Y_prediction))

# 2. Objects with features outside the admissible range must not reach the classifier
test_object = np.array(list(map(float, SAMPLE_OBJECT.split(",")))).reshape(1, -1)
print(test_object)
if is_anomaly(test_object):
    print("Anomalous object: it will not be passed to the classifier.")
    sys.exit()
print("Object is within the admissible range.")

# 3. Classify the object and show the class probabilities
prediction = gnb.predict(test_object)
probabilities = gnb.predict_proba(test_object)
print("Prediction:", prediction, "with probabilities", probabilities)
