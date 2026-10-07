"""
Lesson 3, part B: the multivariate Gaussian and conditioning.

Run from the project folder:
    python 03_math_you_will_lean_on/b_gaussian_conditioning.py

The one-sentence idea:
    A Gaussian over several numbers is a smooth hill. Once you learn the value of some of the
    numbers, you slice the hill there, and the slice is again a Gaussian, with a shifted centre
    and a smaller spread. That slicing is called conditioning.

Pictures saved into 03_math_you_will_lean_on/outputs/
    b1_gaussian_shapes.png         the covariance decides the shape of the hill
    b2_slicing_the_hill.png        observe one variable, the guess for the other gets sharper
    b3_real_pixels.png             two real neighbouring pixels, two far-apart pixels, and a correlation map
    b4_fill_in_missing_half.png    fill in the hidden bottom half of X-rays using only a Gaussian
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
rng = np.random.default_rng(0)


# ---------- Two helpers used throughout ----------
def gaussian_density(points, mean, cov):
    """Height of the 2D Gaussian hill at each point. points has shape (N, 2)."""
    d = points - mean
    inv = np.linalg.inv(cov)
    exponent = -0.5 * np.einsum("ni,ij,nj->n", d, inv, d)    # the squared "distance" from the centre, accounting for shape
    height = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    return height * np.exp(exponent)


def condition(mean, cov, observed_idx, observed_values):
    """
    The conditioning formula. Works for any number of variables.

    You know the overall mean and covariance of all variables.
    You now observe the values of some of them (observed_idx).
    This returns the mean and covariance of the remaining ones, given what you saw.

        new_mean = mu_m + S_mo @ inv(S_oo) @ (x_o - mu_o)
        new_cov  = S_mm - S_mo @ inv(S_oo) @ S_om
    where o = observed, m = missing.
    """
    all_idx = np.arange(len(mean))
    missing_idx = np.setdiff1d(all_idx, observed_idx)
    mu_o, mu_m = mean[observed_idx], mean[missing_idx]
    S_oo = cov[np.ix_(observed_idx, observed_idx)]
    S_mo = cov[np.ix_(missing_idx, observed_idx)]
    S_mm = cov[np.ix_(missing_idx, missing_idx)]
    gain = np.linalg.solve(S_oo, S_mo.T).T       # same as S_mo @ inv(S_oo), but more stable to compute
    new_mean = mu_m + gain @ (observed_values - mu_o)
    new_cov = S_mm - gain @ S_mo.T
    return new_mean, new_cov, missing_idx


def draw_contours(ax, mean, cov, color="black", levels=4):
    """Draw a few contour lines of the hill."""
    xs = np.linspace(mean[0] - 3.5 * np.sqrt(cov[0, 0]), mean[0] + 3.5 * np.sqrt(cov[0, 0]), 200)
    ys = np.linspace(mean[1] - 3.5 * np.sqrt(cov[1, 1]), mean[1] + 3.5 * np.sqrt(cov[1, 1]), 200)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    z = gaussian_density(pts, mean, cov).reshape(gx.shape)
    ax.contour(gx, gy, z, levels=levels, colors=color, linewidths=1)


# =====================================================================
# Figure B1: the covariance decides the shape of the hill
# =====================================================================
print("=== Figure B1 ===")
shapes = [
    ("independent, equal spread\ncov = [[1, 0], [0, 1]]", np.array([[1.0, 0.0], [0.0, 1.0]])),
    ("independent, different spread\ncov = [[2, 0], [0, 0.3]]", np.array([[2.0, 0.0], [0.0, 0.3]])),
    ("related (correlation 0.8)\ncov = [[1, 0.8], [0.8, 1]]", np.array([[1.0, 0.8], [0.8, 1.0]])),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (title, cov) in zip(axes, shapes):
    mean = np.zeros(2)
    samples = rng.multivariate_normal(mean, cov, size=600)
    ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.4, color="tab:blue")
    draw_contours(ax, mean, cov)
    # The axes of the ellipse are the eigenvectors of the covariance (same idea as the SVD in part A).
    values, vectors = np.linalg.eigh(cov)
    for val, vec in zip(values, vectors.T):
        tip = vec * 2 * np.sqrt(val)
        ax.annotate("", xy=tip, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="tab:red", lw=2))
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    print(f"{title.splitlines()[0]}: ellipse axis lengths {2 * np.sqrt(values)}")
plt.suptitle("Three 2D Gaussians. Red arrows: the ellipse axes (eigenvectors of the covariance)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "b1_gaussian_shapes.png", dpi=120)
plt.close()


# =====================================================================
# Figure B2: slicing the hill (conditioning)
# =====================================================================
print()
print("=== Figure B2 ===")
mean = np.zeros(2)
cov = np.array([[1.0, 0.8], [0.8, 1.0]])
seen_x1 = 1.5                                            # we observe x1 = 1.5 and ask about x2
new_mean, new_cov, _ = condition(mean, cov, observed_idx=np.array([0]), observed_values=np.array([seen_x1]))
print(f"Before seeing anything: x2 has mean {mean[1]:.2f} and spread (std) {np.sqrt(cov[1, 1]):.2f}")
print(f"After seeing x1 = {seen_x1}: x2 has mean {new_mean[0]:.2f} and spread (std) {np.sqrt(new_cov[0, 0]):.2f}")
print("By hand: new mean = 0 + 0.8 / 1 * (1.5 - 0) = 1.2,   new variance = 1 - 0.8 * 0.8 / 1 = 0.36")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
samples = rng.multivariate_normal(mean, cov, size=800)
ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.3, color="tab:blue")
draw_contours(ax, mean, cov)
ax.axvline(seen_x1, color="tab:red", lw=2)
ax.plot(seen_x1, new_mean[0], "o", color="tab:red", ms=9)
ax.errorbar(seen_x1, new_mean[0], yerr=2 * np.sqrt(new_cov[0, 0]), color="tab:red", capsize=6, lw=2)
ax.text(seen_x1 + 0.1, 3.2, f"we saw x1 = {seen_x1}", color="tab:red")
ax.text(seen_x1 + 0.3, new_mean[0] - 0.9, "best guess for x2\nwith a 2-std bar", color="tab:red", fontsize=9)
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_title("the hill, and the slice at x1 = 1.5")

ax = axes[1]
grid = np.linspace(-4, 4, 400)
before = np.exp(-0.5 * grid ** 2 / cov[1, 1]) / np.sqrt(2 * np.pi * cov[1, 1])
after = np.exp(-0.5 * (grid - new_mean[0]) ** 2 / new_cov[0, 0]) / np.sqrt(2 * np.pi * new_cov[0, 0])
ax.plot(grid, before, color="tab:blue", lw=2, label=f"x2 before: mean 0, std {np.sqrt(cov[1, 1]):.2f}")
ax.plot(grid, after, color="tab:red", lw=2, label=f"x2 after seeing x1 = 1.5: mean {new_mean[0]:.2f}, std {np.sqrt(new_cov[0, 0]):.2f}")
ax.set_xlabel("x2"); ax.set_ylabel("how likely"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title("the slice is a narrower Gaussian with a moved centre")
plt.suptitle("Conditioning: learning x1 moves and sharpens our belief about x2", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "b2_slicing_the_hill.png", dpi=120)
plt.close()


# =====================================================================
# Figure B3: real pixels from thousands of X-rays
# =====================================================================
print()
print("=== Figure B3 ===")
train = PneumoniaMNIST(split="train", download=True, root=str(DATA_DIR))
X = train.imgs.reshape(len(train), -1).astype(float)   # 4708 rows, 784 pixel columns, values 0..255
anchor = 14 * 28 + 14                                   # pixel at row 14, column 14 (middle of the chest)
pairs = [("neighbour pixel (row 14, col 15)", 14 * 28 + 15), ("far pixel (row 3, col 25)", 3 * 28 + 25)]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, other) in zip(axes[:2], pairs):
    pts = X[:, [anchor, other]]
    mean = pts.mean(axis=0)
    cov = np.cov(pts.T)
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    seen = 100.0                                        # suppose the anchor pixel is 100 (fairly dark)
    new_mean, new_cov, _ = condition(mean, cov, np.array([0]), np.array([seen]))
    print(f"anchor vs {name}: correlation {corr:.2f}. Knowing anchor = 100, the other pixel is "
          f"{new_mean[0]:.0f} +/- {2 * np.sqrt(new_cov[0, 0]):.0f} (before: {mean[1]:.0f} +/- {2 * np.sqrt(cov[1, 1]):.0f}, using 2-std bars)")
    ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.15, color="tab:blue")
    draw_contours(ax, mean, cov, color="black")
    ax.axvline(seen, color="tab:red", lw=2)
    ax.errorbar(seen, new_mean[0], yerr=2 * np.sqrt(new_cov[0, 0]), fmt="o", color="tab:red", capsize=6, lw=2)
    ax.set_xlabel("brightness of anchor pixel (row 14, col 14)")
    ax.set_ylabel(f"brightness of {name}")
    ax.set_title(f"{name}\ncorrelation {corr:.2f}: guess after seeing anchor = 100 is {new_mean[0]:.0f} +/- {2 * np.sqrt(new_cov[0, 0]):.0f}", fontsize=10)
    ax.set_xlim(0, 255); ax.set_ylim(0, 255); ax.grid(alpha=0.3)

# Correlation of the anchor pixel with every other pixel, shown as a picture.
corr_map = np.array([np.corrcoef(X[:, anchor], X[:, j])[0, 1] for j in range(784)]).reshape(28, 28)
ax = axes[2]
im = ax.imshow(corr_map, cmap="RdBu_r", vmin=-1, vmax=1)
ax.plot(14, 14, "k+", ms=12, mew=2)
ax.set_title("how related every pixel is to the anchor (+)\nnearby pixels: strongly, far pixels: weakly", fontsize=10)
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.046, label="correlation")
plt.suptitle("Real pixels behave like a Gaussian: neighbours are related, so one tells you about the other", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "b3_real_pixels.png", dpi=120)
plt.close()


# =====================================================================
# Figure B4: fill in the hidden half of an X-ray with one Gaussian
# =====================================================================
# Treat each X-ray as 784 numbers. Fit one Gaussian to all training X-rays (a mean image and a
# 784 by 784 covariance). Hide the bottom half of a test X-ray and use the conditioning formula
# to guess it from the top half. No neural network, just the formula above.
print()
print("=== Figure B4 ===")
Xs = X / 255.0                                           # work with values 0..1
mean = Xs.mean(axis=0)
cov = np.cov(Xs.T) + 1e-3 * np.eye(784)                  # small extra on the diagonal keeps the math stable
top_idx = np.arange(0, 392)                              # rows 0..13 are the top half
test = PneumoniaMNIST(split="test", download=True, root=str(DATA_DIR))
T = test.imgs.reshape(len(test), -1).astype(float) / 255.0

# The "gain" and the uncertainty do not depend on which image we look at, so compute them once.
_, new_cov, bottom_idx = condition(mean, cov, top_idx, Xs[0, top_idx])
uncertainty = np.sqrt(np.diag(new_cov)).reshape(14, 28)

errors_gauss, errors_avg = [], []
for row in T:
    guess, _, _ = condition(mean, cov, top_idx, row[top_idx])
    errors_gauss.append(np.abs(guess - row[bottom_idx]).mean())
    errors_avg.append(np.abs(mean[bottom_idx] - row[bottom_idx]).mean())
print(f"Average pixel error on {len(T)} test X-rays (0..1 scale):")
print(f"  filling with the average bottom half:    {np.mean(errors_avg):.4f}")
print(f"  filling with the Gaussian conditioning:  {np.mean(errors_gauss):.4f}")

fig = plt.figure(figsize=(14, 9))
grid = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1, 1.3])
show = [0, 5, 11, 17, 23, 29]
for col, i in enumerate(show):
    row = T[i]
    guess, _, _ = condition(mean, cov, top_idx, row[top_idx])
    hidden = row.copy(); hidden[bottom_idx] = 0.5
    filled = row.copy(); filled[bottom_idx] = np.clip(guess, 0, 1)
    for r, (img, title) in enumerate([(row, "true X-ray"), (hidden, "bottom half hidden"), (filled, "filled in by the Gaussian")]):
        ax = fig.add_subplot(grid[r, col])
        ax.imshow(img.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if col == 0:
            ax.text(-3, 14, title, ha="right", va="center", fontsize=10)
ax = fig.add_subplot(grid[3, 1:5])
im = ax.imshow(uncertainty, cmap="magma")
ax.set_title("how unsure the Gaussian is about each hidden pixel (std of the guess)\nthe same map for every image: it depends on the Gaussian, not on the picture", fontsize=10)
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.03, label="std of the guess")
plt.suptitle("Conditioning a 784-number Gaussian: guess the bottom half of an X-ray from the top half", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "b4_fill_in_missing_half.png", dpi=120)
plt.close()

print()
print("Done. Pictures are in:", OUT_DIR)
