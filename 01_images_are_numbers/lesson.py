"""
Lesson 1: An image is just a grid of numbers.

Run from the project folder:
    python 01_images_are_numbers/lesson.py

What happens:
    1. A tiny chest X-ray dataset downloads (about 4 MB) into data/
    2. We look at one image as raw numbers
    3. We save four pictures into 01_images_are_numbers/outputs/
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw into files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
from medmnist import PathMNIST, PneumoniaMNIST

# ---------- Where files live ----------
HERE = Path(__file__).parent              # this lesson's folder
DATA_DIR = HERE.parent / "data"           # downloads go here (not pushed to GitHub)
OUT_DIR = HERE / "outputs"                # pictures we make go here
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


# ---------- Step 1: get the data ----------
# PneumoniaMNIST is a set of children's chest X-rays, shrunk to 28 by 28 pixels.
# Each image has a label: 0 means normal, 1 means pneumonia.
# "split" picks which pile we want: train, val, or test.
train = PneumoniaMNIST(split="train", download=True, root=str(DATA_DIR))
label_names = train.info["label"]         # a small dictionary: {"0": "normal", "1": "pneumonia"}
print("Number of training images:", len(train))
print("Label names:", label_names)
print()


# ---------- Step 2: one image as raw numbers ----------
image, label = train[0]                   # image is a picture object, label is a tiny array like [0]
pixels = np.array(image)                  # turn the picture into a grid of numbers

print("What Python thinks the image is:", type(image).__name__)
print("Shape of the grid (rows, columns):", pixels.shape)
print("Type of each number:", pixels.dtype, "(whole numbers from 0 to 255)")
print("Darkest pixel:", pixels.min(), "  Brightest pixel:", pixels.max())
print("Label:", label[0], "which means:", label_names[str(label[0])])
print()

print("The top-left 8 by 8 corner of the image, as numbers:")
np.set_printoptions(linewidth=200)
print(pixels[:8, :8])
print("(0 is black, 255 is white, in between is grey)")
print()


# ---------- Step 3: draw the image with its numbers on top ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Left: the full image, drawn big
axes[0].imshow(pixels, cmap="gray", vmin=0, vmax=255)
axes[0].set_title(f"Full image, 28 x 28 pixels\nlabel: {label_names[str(label[0])]}")
axes[0].axis("off")

# Right: a zoomed 10 by 10 patch from the middle, with each number written in its cell
patch = pixels[9:19, 9:19]
axes[1].imshow(patch, cmap="gray", vmin=0, vmax=255)
for row in range(patch.shape[0]):
    for col in range(patch.shape[1]):
        value = patch[row, col]
        text_color = "black" if value > 128 else "white"   # keep the text readable
        axes[1].text(col, row, str(value), ha="center", va="center", color=text_color, fontsize=8)
axes[1].set_title("Zoomed 10 x 10 patch from the middle\neach cell shows its pixel value")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(OUT_DIR / "01_pixel_values.png", dpi=120)
plt.close()


# ---------- Step 4: a grid of images with labels ----------
fig, axes = plt.subplots(3, 6, figsize=(12, 6.5))
for i, ax in enumerate(axes.flat):
    img, lab = train[i]
    ax.imshow(np.array(img), cmap="gray", vmin=0, vmax=255)
    ax.set_title(label_names[str(lab[0])], fontsize=10)
    ax.axis("off")
plt.suptitle("18 chest X-rays from the training pile")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_sample_grid.png", dpi=120)
plt.close()


# ---------- Step 5: a color image has three layers ----------
# PathMNIST is colored microscope pictures of colon tissue.
color_train = PathMNIST(split="train", download=True, root=str(DATA_DIR))
color_image, color_label = color_train[0]
color_pixels = np.array(color_image)
print("Color image shape (rows, columns, channels):", color_pixels.shape)
print("The 3 at the end means three layers: red, green, blue")
print("Pixel at row 0, column 0 is (red, green, blue) =", color_pixels[0, 0])
print()

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(color_pixels)
axes[0].set_title("Full color image\n" + color_train.info["label"][str(color_label[0])])
for i, (name, cmap) in enumerate(zip(["Red layer", "Green layer", "Blue layer"], ["Reds", "Greens", "Blues"])):
    axes[i + 1].imshow(color_pixels[:, :, i], cmap=cmap, vmin=0, vmax=255)
    axes[i + 1].set_title(name)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_color_channels.png", dpi=120)
plt.close()


# ---------- Step 6: how many of each label ----------
# train.labels is a list of all the answers. We count how often each one appears.
all_labels = train.labels.flatten()
counts = {label_names[str(k)]: int((all_labels == k).sum()) for k in np.unique(all_labels)}
print("How many training images of each kind:", counts)
print("Notice one kind is about 3 times more common. We will deal with that in Lesson 8.")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(list(counts.keys()), list(counts.values()), color=["tab:green", "tab:red"])
for i, v in enumerate(counts.values()):
    ax.text(i, v + 30, str(v), ha="center")
ax.set_ylabel("number of images")
ax.set_title("Training images per label")
plt.tight_layout()
plt.savefig(OUT_DIR / "04_label_counts.png", dpi=120)
plt.close()

print()
print("Done. Open the pictures in:", OUT_DIR)
