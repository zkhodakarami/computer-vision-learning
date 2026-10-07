"""
Lesson 2: Changing images.

Run from the project folder:
    python 02_changing_images/lesson.py

What happens:
    1. A 64 by 64 version of the chest X-ray dataset downloads (about 20 MB) into data/
    2. One X-ray goes through eleven common operations
    3. Two pictures are saved into 02_changing_images/outputs/
"""

from pathlib import Path

import cv2                                 # OpenCV: the classic image toolbox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from medmnist import PneumoniaMNIST

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"
OUT_DIR = HERE / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


# ---------- Step 1: get one image ----------
# size=64 asks for the 64 by 64 pixel version instead of the tiny 28 by 28 one.
train = PneumoniaMNIST(split="train", download=True, root=str(DATA_DIR), size=64)
image, label = train[3]
pixels = np.array(image)                   # grid of whole numbers from 0 to 255
height, width = pixels.shape
print("Image shape:", pixels.shape, "  label:", train.info["label"][str(label[0])])
print()


# ---------- Step 2: the operations ----------
# Every result goes into this dictionary so we can draw them all at the end.
results = {}
results["original"] = pixels

# Resize bigger. INTER_NEAREST copies the closest pixel, so it looks blocky.
results["resize to 256 (nearest)"] = cv2.resize(pixels, (256, 256), interpolation=cv2.INTER_NEAREST)

# Same resize but INTER_LINEAR blends neighbours, so it looks smooth.
results["resize to 256 (linear)"] = cv2.resize(pixels, (256, 256), interpolation=cv2.INTER_LINEAR)

# Resize smaller. Detail is lost and cannot be recovered.
results["resize to 16"] = cv2.resize(pixels, (16, 16), interpolation=cv2.INTER_AREA)

# Crop. Just keep the middle 40 by 40 block of the grid.
results["crop middle"] = pixels[12:52, 12:52]

# Flip left to right. The column order is reversed.
results["flip left-right"] = np.fliplr(pixels)

# Rotate 15 degrees around the centre. Empty corners get filled with black.
turn = cv2.getRotationMatrix2D(center=(width / 2, height / 2), angle=15, scale=1.0)
results["rotate 15 degrees"] = cv2.warpAffine(pixels, turn, (width, height))

# Brighter. Add 60 to every pixel, then make sure nothing goes above 255.
results["brighter (+60)"] = np.clip(pixels.astype(int) + 60, 0, 255).astype(np.uint8)

# More contrast. Push values away from middle grey (128).
results["more contrast (x1.6)"] = np.clip((pixels.astype(float) - 128) * 1.6 + 128, 0, 255).astype(np.uint8)

# Blur. Each pixel becomes a weighted average of its 5 by 5 neighbourhood.
results["blur (5x5)"] = cv2.GaussianBlur(pixels, (5, 5), 0)

# Edges. White where brightness changes sharply, black elsewhere.
results["edges (Canny)"] = cv2.Canny(pixels, 50, 150)

# Equalize. Spread the brightness values out so dark and bright areas both show detail.
results["equalize histogram"] = cv2.equalizeHist(pixels)

# Draw all twelve in a grid.
fig, axes = plt.subplots(3, 4, figsize=(13, 10))
for ax, (name, img) in zip(axes.flat, results.items()):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{name}\nshape {img.shape}", fontsize=10)
    ax.axis("off")
plt.suptitle("One X-ray, eleven common operations")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_operations.png", dpi=120)
plt.close()


# ---------- Step 3: what "normalize" means ----------
# Pixel values go from 0 to 255. Neural networks train much better when the numbers are small.
scaled = pixels / 255.0                                     # now from 0.0 to 1.0
standardized = (scaled - scaled.mean()) / scaled.std()      # now centred on 0, spread of about 1

print("Same image, three ways of writing the numbers:")
print(f"  raw          : smallest {pixels.min():>7}, largest {pixels.max():>7}, average {pixels.mean():>7.2f}")
print(f"  scaled 0..1  : smallest {scaled.min():>7.3f}, largest {scaled.max():>7.3f}, average {scaled.mean():>7.3f}")
print(f"  standardized : smallest {standardized.min():>7.3f}, largest {standardized.max():>7.3f}, average {standardized.mean():>7.3f}")
print("All three look identical when drawn. Only the numbers changed.")
print()


# ---------- Step 4: histograms, before and after equalizing ----------
# A histogram counts how many pixels have each brightness value.
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for col, (name, img) in enumerate([("original", pixels), ("equalized", results["equalize histogram"])]):
    axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0, col].set_title(name)
    axes[0, col].axis("off")
    axes[1, col].hist(img.flatten(), bins=64, range=(0, 255), color="steelblue")
    axes[1, col].set_xlabel("pixel brightness (0 = black, 255 = white)")
    axes[1, col].set_ylabel("how many pixels")
plt.suptitle("Equalizing spreads the brightness values across the whole range")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_histograms.png", dpi=120)
plt.close()

print("Done. Open the pictures in:", OUT_DIR)
