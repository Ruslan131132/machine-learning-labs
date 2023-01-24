from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import argrelmin

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
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
