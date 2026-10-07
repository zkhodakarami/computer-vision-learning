"""
Lesson 3, part A: the singular value decomposition (SVD), range and null space.

Run from the project folder:
    python 03_math_you_will_lean_on/a_svd_range_null_space.py

The one-sentence idea:
    Any matrix, no matter how messy, is just "rotate, stretch along a few directions, rotate again".
    The SVD finds those directions and how much each one is stretched.

Pictures saved into 03_math_you_will_lean_on/outputs/
    a1_circle_to_ellipse.png     what a 2 by 2 matrix does to a circle, range and null space
    a2_xray_compression.png      an X-ray rebuilt from only its biggest stretch directions
    a3_main_directions.png       the directions in which thousands of X-rays differ most
    a4_blur_null_space.png       why you cannot un-blur: blur almost erases wiggly patterns
"""

from pathlib import Path

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
np.set_printoptions(precision=3, suppress=True)


# =====================================================================
# Figure A1: what a matrix does to a circle
# =====================================================================
# A matrix is a machine that takes a vector in and gives a vector out.
# Feed it every point on a circle and look at what comes out.

theta = np.linspace(0, 2 * np.pi, 400)
circle = np.vstack([np.cos(theta), np.sin(theta)])        # shape (2, 400): 400 points on the unit circle

A = np.array([[2.0, 1.0],
              [0.5, 1.5]])                                 # a healthy matrix: nothing gets erased
B = np.array([[1.0, 2.0],
              [0.5, 1.0]])                                 # a flat matrix: row 2 is half of row 1

print("=== Figure A1 ===")
for name, M in [("A", A), ("B", B)]:
    # np.linalg.svd gives U, the singular values s, and V transposed.
    # M = U @ diag(s) @ Vt.  Columns of V are the input directions, columns of U the output directions.
    U, s, Vt = np.linalg.svd(M)
    print(f"Matrix {name}:")
    print(f"  singular values (how much each direction is stretched): {s}")
    print(f"  rank (number of non-zero singular values): {np.linalg.matrix_rank(M)}")

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for row, (name, M) in enumerate([("A: healthy matrix (rank 2)", A), ("B: flat matrix (rank 1)", B)]):
    U, s, Vt = np.linalg.svd(M)
    V = Vt.T
    out = M @ circle                                       # where every circle point lands

    # Left panel: the input side
    ax = axes[row, 0]
    ax.plot(circle[0], circle[1], color="gray")
    for i, color in enumerate(["tab:blue", "tab:orange"]):
        ax.annotate("", xy=V[:, i], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=2))
        ax.text(*(V[:, i] * 1.15), f"v{i + 1}", color=color, fontsize=12, ha="center")
    if s[1] < 1e-10:
        # the direction v2 is squashed to zero: that is the null space
        line = np.outer(V[:, 1], [-3, 3])
        ax.plot(line[0], line[1], "--", color="tab:orange", lw=1.5)
        ax.text(-2.4, 1.9, "null space:\nevery input on this line\ncomes out as 0", color="tab:orange", fontsize=9)
    ax.set_title(f"{name}\ninput: the unit circle and the input directions v1, v2")
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.grid(alpha=0.3)

    # Right panel: the output side
    ax = axes[row, 1]
    ax.plot(out[0], out[1], color="black")
    for i, color in enumerate(["tab:blue", "tab:orange"]):
        tip = s[i] * U[:, i]
        if s[i] > 1e-10:
            ax.annotate("", xy=tip, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=2))
            ax.text(*(tip * 1.12), f"{s[i]:.2f} x u{i + 1}", color=color, fontsize=11, ha="center")
    if s[1] < 1e-10:
        line = np.outer(U[:, 0], [-3, 3])
        ax.plot(line[0], line[1], ":", color="tab:blue", lw=1)
        ax.text(-2.8, 2.2, "range:\nall outputs land\non this one line", color="tab:blue", fontsize=9)
        ax.set_title("output: the circle is squashed flat onto a line segment")
    else:
        ax.set_title("output: the circle becomes an ellipse\naxis lengths = singular values, axis directions = u1, u2")
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.grid(alpha=0.3)

plt.suptitle("SVD: a matrix = rotate (V), stretch along each axis (singular values), rotate (U)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "a1_circle_to_ellipse.png", dpi=120)
plt.close()


# =====================================================================
# Figure A2: an X-ray rebuilt from only its biggest directions
# =====================================================================
# An image is a matrix, so it has an SVD. Keep only the k biggest singular values
# and you get a compressed version. This is the idea behind many compression tricks.

train64 = PneumoniaMNIST(split="train", download=True, root=str(DATA_DIR), size=64)
xray = np.array(train64[3][0]).astype(float)              # 64 by 64 grid of numbers
U, s, Vt = np.linalg.svd(xray)

print()
print("=== Figure A2 ===")
print("X-ray shape:", xray.shape, " number of singular values:", len(s))
print("First 5 singular values:", s[:5])
print("Last 5 singular values: ", s[-5:])
print("The first few are huge, the rest are tiny. The tiny ones barely matter.")

ks = [1, 3, 5, 10, 20, 64]
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, k in zip(axes.flat[:6], ks):
    # Rebuild using only the first k pieces. Each piece is one column of U, one value, one row of Vt.
    rebuilt = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    stored = k * (64 + 64 + 1)
    ax.imshow(rebuilt, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"k = {k}\nstores {stored} numbers instead of 4096", fontsize=10)
    ax.axis("off")

ax = axes.flat[6]
ax.plot(np.arange(1, 65), s, "o-", ms=3)
ax.set_yscale("log")
ax.set_xlabel("which singular value (1 = biggest)")
ax.set_ylabel("size (log scale)")
ax.set_title("the singular values drop off fast")
ax.grid(alpha=0.3)

ax = axes.flat[7]
errors = [np.linalg.norm(xray - U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]) / np.linalg.norm(xray) for k in range(1, 65)]
ax.plot(np.arange(1, 65), errors, "o-", ms=3, color="tab:red")
ax.set_xlabel("how many pieces kept (k)")
ax.set_ylabel("relative error")
ax.set_title("error of the rebuilt image")
ax.grid(alpha=0.3)

plt.suptitle("Rebuilding an X-ray from its k biggest SVD pieces", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "a2_xray_compression.png", dpi=120)
plt.close()


# =====================================================================
# Figure A3: the directions in which many X-rays differ most
# =====================================================================
# Put 3000 X-rays side by side as rows of one big matrix (3000 rows, 784 columns).
# The SVD of that matrix finds the "directions" (patterns of 784 pixels) along which the images
# differ the most. This is called principal component analysis (PCA). Same math, new use.

train28 = PneumoniaMNIST(split="train", download=True, root=str(DATA_DIR))
n = 3000
X = train28.imgs[:n].reshape(n, -1).astype(float) / 255.0  # each row is one flattened image
labels = train28.labels[:n].flatten()
mean_image = X.mean(axis=0)
Xc = X - mean_image                                        # centre the data: subtract the average image

U, s, Vt = np.linalg.svd(Xc, full_matrices=False)          # Vt rows are the main directions
coords = Xc @ Vt[:2].T                                     # each image's position along the first two directions

print()
print("=== Figure A3 ===")
print("Data matrix shape (images, pixels):", X.shape)
print("Share of total variation captured by the first 2 directions:",
      f"{(s[:2] ** 2).sum() / (s ** 2).sum():.1%}", " by the first 20:", f"{(s[:20] ** 2).sum() / (s ** 2).sum():.1%}")

fig = plt.figure(figsize=(14, 7))
grid = fig.add_gridspec(2, 6)
ax = fig.add_subplot(grid[0, 0])
ax.imshow(mean_image.reshape(28, 28), cmap="gray")
ax.set_title("average X-ray", fontsize=10); ax.axis("off")
for i in range(5):
    ax = fig.add_subplot(grid[0, i + 1])
    ax.imshow(Vt[i].reshape(28, 28), cmap="RdBu_r")
    ax.set_title(f"direction {i + 1}\nstretch {s[i]:.0f}", fontsize=10); ax.axis("off")
ax = fig.add_subplot(grid[1, :3])
for value, name, color in [(0, "normal", "tab:green"), (1, "pneumonia", "tab:red")]:
    pick = labels == value
    ax.scatter(coords[pick, 0], coords[pick, 1], s=6, alpha=0.4, color=color, label=name)
ax.set_xlabel("position along direction 1"); ax.set_ylabel("position along direction 2")
ax.set_title("each dot is one X-ray, placed by its first two coordinates"); ax.legend(); ax.grid(alpha=0.3)
ax = fig.add_subplot(grid[1, 3:])
ax.plot(np.arange(1, 51), s[:50] ** 2 / (s ** 2).sum() * 100, "o-", ms=3)
ax.set_xlabel("direction number"); ax.set_ylabel("% of variation explained")
ax.set_title("a handful of directions explain most of the differences"); ax.grid(alpha=0.3)
plt.suptitle("SVD of a stack of 3000 X-rays: the main ways they differ (PCA)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "a3_main_directions.png", dpi=120)
plt.close()


# =====================================================================
# Figure A4: why you cannot un-blur, explained by a near null space
# =====================================================================
# Blurring a 1D signal of 64 numbers is a matrix (64 by 64): each output is a weighted
# average of its neighbours. Look at that matrix's SVD: some singular values are almost zero.
# The input directions that go with them are wiggly patterns. Blur nearly erases them.
# Once erased, they cannot be brought back. That is why sharpening a blurry photo is so hard.

size = 64
weights = np.array([1, 4, 6, 4, 1]) / 16.0                 # a simple blur: weighted average of 5 neighbours
Blur = np.zeros((size, size))
for i in range(size):
    for offset, w in zip(range(-2, 3), weights):
        j = i + offset
        if 0 <= j < size:
            Blur[i, j] = w
U, s, Vt = np.linalg.svd(Blur)

print()
print("=== Figure A4 ===")
print("Blur matrix singular values, biggest:", s[:3], " smallest:", s[-3:])
print("The smallest are almost 0, so those input patterns are almost erased.")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
axes[0].imshow(Blur, cmap="viridis")
axes[0].set_title("the blur as a 64 by 64 matrix\n(each row: weights of the 5 neighbours)")
axes[1].semilogy(np.arange(1, size + 1), s, "o-", ms=3)
axes[1].set_xlabel("which singular value"); axes[1].set_ylabel("size (log scale)")
axes[1].set_title("singular values of the blur\nthe last ones are nearly zero"); axes[1].grid(alpha=0.3)
axes[2].plot(Vt[0], label=f"direction 1 (kept, stretch {s[0]:.2f})", color="tab:blue")
axes[2].plot(Vt[-1], label=f"direction 64 (erased, stretch {s[-1]:.4f})", color="tab:red", alpha=0.8)
axes[2].set_title("smooth patterns survive the blur,\nwiggly patterns are almost erased")
axes[2].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2); axes[2].grid(alpha=0.3)
plt.suptitle("Near null space: the blur almost erases wiggly input patterns, so un-blurring is ill-posed", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "a4_blur_null_space.png", dpi=120)
plt.close()

print()
print("Done. Pictures are in:", OUT_DIR)
