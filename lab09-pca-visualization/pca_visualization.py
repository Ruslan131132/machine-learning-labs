from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_PATH = DATA_DIR / "new_data.csv"

TARGET = "attack"

data_frame = pd.read_csv(DATASET_PATH, index_col=0)
X = data_frame.drop(TARGET, axis=1)
Y = data_frame[TARGET]

# 1. Project the data onto the first two principal components
pca = PCA(n_components=2)
principal_df_X = pd.DataFrame(pca.fit_transform(X), columns=["PC1", "PC2"])

plt.figure()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("PC1", fontsize=14)
plt.ylabel("PC2", fontsize=14)
plt.title("Data in the space of the first two principal components", fontsize=10)
plt.scatter(principal_df_X["PC1"], principal_df_X["PC2"])
plt.show()

# The same projection colored by class: 0 - normal traffic, 1 - attack
plt.scatter(principal_df_X["PC1"], principal_df_X["PC2"], c=Y, s=20)
plt.show()

# 2. Heatmap of feature loadings for the first two principal components
ax = sns.heatmap(
    pca.components_,
    yticklabels=[f"PC{i}" for i in range(1, pca.n_components_ + 1)],
    xticklabels=list(X.columns),
    cbar_kws={"orientation": "horizontal"},
)
ax.set_aspect("equal")
plt.show()

# 3. Features with near-zero loadings in both components are the least informative
loadings = pd.DataFrame(pca.components_, index=["PC1", "PC2"], columns=X.columns)
print("Features ordered from least to most informative:")
print(loadings.abs().max().sort_values())
