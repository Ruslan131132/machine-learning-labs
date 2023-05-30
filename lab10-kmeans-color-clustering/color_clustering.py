from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COLORS_PATH = DATA_DIR / "colors_2.csv"


def plot_elbow(data, k_range, title):
    """Plot k-means inertia for every k; the optimal k is at the 'elbow' of the curve."""
    distortions = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10)
        model.fit(data)
        distortions.append(model.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, distortions, "bx-")
    plt.xlabel("k")
    plt.ylabel("Distortion")
    plt.title(f"The Elbow Method showing the optimal k ({title})")
    plt.show()


data = pd.read_csv(COLORS_PATH, sep=":").dropna()
data.rename(columns={"x1": "r", "x2": "g", "x3": "b"}, inplace=True)

plot_elbow(data, range(1, 15), "RGB")
