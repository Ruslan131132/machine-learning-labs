from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DATASET_PATH = DATA_DIR / "dataframe.csv"
PROCESSED_DATASET_PATH = DATA_DIR / "new_data.csv"

TARGET = "attack"
LOW_INFORMATION_SHARE = 0.90
Z_SCORE_LIMIT = 3

data_frame = pd.read_csv(RAW_DATASET_PATH)

# Protocol frequency chart: port values depend on the protocol, so it drives the cleanup below
sns.set(color_codes=True)
plot = sns.displot(data_frame["proto"].dropna())
plot.fig.set_size_inches(7, 7)
plt.show()
print(data_frame.proto.unique())

# ARP rows have no ports and ICMP rows store hex codes instead of ports; both are a small share of the data.
# Two more rows contain service names ("nut", "xinetd") instead of port numbers.
data_frame = data_frame[data_frame.proto != "arp"]
data_frame = data_frame[data_frame.proto != "icmp"]
data_frame = data_frame[data_frame.dport != "nut"]
data_frame = data_frame[data_frame.sport != "xinetd"]

# 1. Rows with missing values cannot be dropped directly: some columns are empty in 100% of rows,
# so report the share of missing values per column first
for col in data_frame.columns:
    pct_missing = np.mean(data_frame[col].isnull())
    print(f"{col} - {round(pct_missing * 100)}%")

numeric_cols = data_frame.select_dtypes(include=[np.number]).columns.values
non_numeric_cols = data_frame.select_dtypes(exclude=[np.number]).columns.values
print("Numeric columns:", numeric_cols)
print("Non-numeric columns:", non_numeric_cols)

# 2. Drop uninformative features
# Columns that are empty in every row
data_frame.drop(["smac", "dmac", "soui", "doui", "sco", "dco"], axis=1, inplace=True)
# Identifiers, timestamps and transaction state carry no signal for attack detection
data_frame.drop(["pkSeqID", "stime", "ltime", "flgs", "state", "seq"], axis=1, inplace=True)

# Features where a single value covers more than 90% of rows
num_rows = len(data_frame.index)
low_information_cols = []
for col in data_frame.columns:
    counts = data_frame[col].value_counts(dropna=False)
    top_pct = (counts / num_rows).iloc[0]
    if top_pct > LOW_INFORMATION_SHARE:
        low_information_cols.append(col)
        print(f"{col}: {top_pct * 100:.5f}%")
        print(counts)
        print()

# Text categories are almost constant and only refine the target label
data_frame.drop(["category", "subcategory "], axis=1, inplace=True)
# Protocol, addresses and ports identify a connection rather than describe its behavior
data_frame.drop(["proto", "saddr", "daddr", "sport", "dport"], axis=1, inplace=True)

# 3. All remaining features are numeric, so no categorical encoding is required

# 4. Standardize features to zero mean and unit variance; the binary target is left as is
feature_cols = data_frame.columns.drop(TARGET)
scaler = StandardScaler().fit(data_frame[feature_cols].values)
data_frame[feature_cols] = scaler.transform(data_frame[feature_cols].values)

# 5. Drop rows with outliers: |z| > 3 in any feature
for col in feature_cols:
    data_frame.loc[(data_frame[col] > Z_SCORE_LIMIT) | (data_frame[col] < -Z_SCORE_LIMIT), col] = None
data_frame = data_frame.dropna()

# 6. Balance classes by downsampling the majority class (attacks) to the size of the minority class
data_majority = data_frame[data_frame[TARGET] == 1]
data_minority = data_frame[data_frame[TARGET] == 0]
data_majority_downsampled = resample(
    data_majority,
    replace=False,
    n_samples=len(data_minority),
    random_state=600,
)
data_frame = pd.concat([data_majority_downsampled, data_minority])

data_frame.to_csv(PROCESSED_DATASET_PATH)
print(f"Saved {len(data_frame)} rows to {PROCESSED_DATASET_PATH}")
