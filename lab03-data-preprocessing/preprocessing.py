from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DATASET_PATH = DATA_DIR / "dataframe.csv"

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
