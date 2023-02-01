from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import argrelmin

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
IMAGE_PATH = DATA_DIR / "plane.jpg"

# ITU-R BT.601 luma coefficients for RGB -> grayscale conversion
LUMA_WEIGHTS = [0.299, 0.587, 0.114]
BINS = 54

img = np.asarray(Image.open(IMAGE_PATH))
intensity_img = np.average(img, axis=-1, weights=LUMA_WEIGHTS)

# Brightness histogram and its density curve
density, bin_edges, _ = plt.hist(intensity_img.ravel(), bins=BINS)
values = bin_edges[:-1] + np.diff(bin_edges)[0] / 2
plt.plot(values, density)
normal_density = density / density.sum()

# Local minima of the density split the histogram into homogeneous brightness regions
local_minima = [values[i] for i in argrelmin(normal_density)[0]]
for minimum in local_minima:
    plt.axvline(minimum, color="green", ls=":")
plt.show()

# Otsu-style threshold search: minimize the total spread of the two pixel groups
freqs, _ = np.histogram(intensity_img.ravel(), bins=BINS)
best_loss = 100
threshold = 0
for i in range(len(freqs)):
    current_freq = freqs[i]
    group1 = intensity_img[intensity_img < current_freq]
    group2 = intensity_img[intensity_img >= current_freq]
    loss = group1.std() + group2.std()
    if loss < best_loss:
        best_loss = loss
        threshold = i

# Binary mask: dark pixels (the plane body) become 255, the background becomes 0
segmented_img = intensity_img.copy()
segmented_img[intensity_img > threshold] = 0
segmented_img[intensity_img < threshold] = 255

# Highlight the segmented region in red
new_img = img.copy()
x, y = np.where(segmented_img == 255)
new_img[x, y, 0] = 240
plt.imshow(new_img)
plt.show()

# Label every pixel by the brightness region bounded by the 3rd and 5th local minima:
# 1 - all channels are dark, 2 - all channels are in the middle range, 3 - everything else
low, high = local_minima[2], local_minima[4]
pixels = img.transpose(1, 0, 2).reshape(-1, 3).astype(int)
labels = np.full(len(pixels), 3)
labels[np.all((pixels > low) & (pixels <= high), axis=1)] = 2
labels[np.all(pixels <= low, axis=1)] = 1

OUTPUT_DIR.mkdir(exist_ok=True)
result = pd.DataFrame({"R": pixels[:, 0], "G": pixels[:, 1], "B": pixels[:, 2], "label": labels})
result.to_csv(OUTPUT_DIR / "rgbl_result.csv")
