import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
COLORS_PATH = DATA_DIR / "colors_2.csv"
HTML_PATH = OUTPUT_DIR / "lab10_colors_2.html"

LUMA_WEIGHTS = [0.299, 0.587, 0.114]

# Picked from the elbow plots
RGB_CLUSTERS = 6
HSV_CLUSTERS = 4
GRAY_CLUSTERS = 4


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


def to_hex(c1, c2, c3):
    return "#{:02x}{:02x}{:02x}".format(int(c1), int(c2), int(c3))


def build_cluster_table(caption, centers, points, labels):
    """Render an HTML table with one row per cluster: the centroid cell first, then every object of the cluster."""
    html = f"<table border='1'>\n<caption>{caption}</caption>\n<tr><th>Centroid</th></tr>\n"
    for cluster, center in enumerate(centers):
        html += f"<tr><td bgcolor='{to_hex(*center)}' height=25 width=25></td>"
        for point in points[labels == cluster]:
            html += f'<td bgcolor="{to_hex(*point)}" height=25 width=25></td>'
        html += "</tr>\n"
    html += "</table>\n"
    return html


data = pd.read_csv(COLORS_PATH, sep=":").dropna()
data.rename(columns={"x1": "r", "x2": "g", "x3": "b"}, inplace=True)
rgb_points = data[["r", "g", "b"]].to_numpy()

plot_elbow(rgb_points, range(1, 15), "RGB")
kmeans = KMeans(n_clusters=RGB_CLUSTERS, n_init=10)
kmeans.fit(rgb_points)
result = build_cluster_table("RGB", kmeans.cluster_centers_, rgb_points, kmeans.labels_)

# Repeat the clustering with colors expressed in the HSV model
hsv_rows = []
for r, g, b in rgb_points:
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    # OpenCV-style ranges: H in [0, 179], S and V in [0, 255]
    hsv_rows.append([int(h * 179), int(s * 255), int(v * 255)])
hsv_points = np.array(hsv_rows)

plot_elbow(hsv_points, range(1, 15), "HSV")
kmeans_hsv = KMeans(n_clusters=HSV_CLUSTERS, n_init=10)
kmeans_hsv.fit(hsv_points)
result += build_cluster_table("HSV", kmeans_hsv.cluster_centers_, hsv_points, kmeans_hsv.labels_)

# ...and in grayscale, using the same luma weights as in lab 1
gray_points = rgb_points @ np.array(LUMA_WEIGHTS)
gray_points = gray_points.reshape(-1, 1)

plot_elbow(gray_points, range(1, 10), "gray")
kmeans_gray = KMeans(n_clusters=GRAY_CLUSTERS, n_init=10)
kmeans_gray.fit(gray_points)
result += build_cluster_table(
    "Gray",
    kmeans_gray.cluster_centers_.repeat(3, axis=1),
    gray_points.repeat(3, axis=1),
    kmeans_gray.labels_,
)

OUTPUT_DIR.mkdir(exist_ok=True)
HTML_PATH.write_text(result, encoding="utf-8")
