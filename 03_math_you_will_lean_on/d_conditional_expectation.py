"""
Lesson 3, part D: conditional expectation, the best guess.

Run from the project folder:
    python 03_math_you_will_lean_on/d_conditional_expectation.py

The one-sentence idea:
    The conditional expectation E[Y | X] is "the average of Y among the cases where X looks like this".
    It is the single best guess for Y when you know X, if "best" means smallest squared error.
    Every model trained with squared error is trying to learn this curve.

Pictures saved into 03_math_you_will_lean_on/outputs/
    d1_best_guess_curve.png      brightness of the bottom half of an X-ray, guessed from the top half
    d2_why_the_average.png       the average is the guess with the smallest squared error
    d3_probability_curve.png     with yes or no labels, the best guess is a probability
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

train = PneumoniaMNIST(split="train", download=True, root=str(DATA_DIR))
images = train.imgs.astype(float)                       # 4708 images, each 28 by 28, values 0..255
labels = train.labels.flatten()                         # 0 = normal, 1 = pneumonia


def binned_average(x, y, n_bins=15):
    """
    Cut the x axis into bins that each hold about the same number of points.
    Inside each bin, take the average of y. That average IS the conditional expectation E[Y | X in bin].
    Returns bin centres, averages, spreads (std) and counts.
    """
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
    centres = np.array([x[which == b].mean() for b in range(n_bins)])
    averages = np.array([y[which == b].mean() for b in range(n_bins)])
    spreads = np.array([y[which == b].std() for b in range(n_bins)])
    counts = np.array([(which == b).sum() for b in range(n_bins)])
    return centres, averages, spreads, counts, which


# =====================================================================
# Figure D1: guess the bottom half's brightness from the top half's
# =====================================================================
print("=== Figure D1 ===")
x = images[:, :14, :].mean(axis=(1, 2))                 # X: average brightness of the top half
y = images[:, 14:, :].mean(axis=(1, 2))                 # Y: average brightness of the bottom half
centres, averages, spreads, counts, which = binned_average(x, y)

# If X and Y were jointly Gaussian, E[Y | X] would be a straight line with slope cov / var.
slope = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
intercept = y.mean() - slope * x.mean()
print(f"Spread of Y on its own (std): {y.std():.1f}")
print(f"Typical spread of Y once X is known (average std inside bins): {spreads.mean():.1f}")
print(f"Straight-line guess: Y = {slope:.2f} * X + {intercept:.1f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
ax.scatter(x, y, s=4, alpha=0.2, color="tab:blue", label="one X-ray each")
ax.errorbar(centres, averages, yerr=spreads, fmt="o-", color="tab:red", capsize=3, lw=2, label="E[Y | X]: average Y in each X bin, with 1-std bars")
line_x = np.array([x.min(), x.max()])
ax.plot(line_x, slope * line_x + intercept, "--", color="black", lw=1.5, label="straight-line guess (what a Gaussian would say)")
ax.set_xlabel("X: average brightness of the top half"); ax.set_ylabel("Y: average brightness of the bottom half")
ax.set_title("the best-guess curve"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.hist(y, bins=40, color="lightgray", density=True, label=f"all Y (std {y.std():.0f})")
for b, color in zip([1, 7, 13], ["tab:green", "tab:orange", "tab:purple"]):
    pick = which == b
    ax.hist(y[pick], bins=20, density=True, alpha=0.5, color=color,
            label=f"Y when X is about {centres[b]:.0f} (std {spreads[b]:.0f})")
ax.set_xlabel("Y: average brightness of the bottom half"); ax.set_ylabel("how common")
ax.set_title("knowing X narrows down Y"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle("Conditional expectation = the average of Y among cases with similar X", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "d1_best_guess_curve.png", dpi=120)
plt.close()


# =====================================================================
# Figure D2: why the average is the best guess
# =====================================================================
print()
print("=== Figure D2 ===")
pick = which == 7                                        # all X-rays in the middle bin
y_bin = y[pick]
guesses = np.linspace(y_bin.min(), y_bin.max(), 400)
squared_error = np.array([np.mean((y_bin - g) ** 2) for g in guesses])
absolute_error = np.array([np.mean(np.abs(y_bin - g)) for g in guesses])
print(f"In the middle bin ({pick.sum()} X-rays): mean of Y = {y_bin.mean():.1f}, median of Y = {np.median(y_bin):.1f}")
print(f"Squared error is smallest at guess = {guesses[squared_error.argmin()]:.1f}  (the mean)")
print(f"Absolute error is smallest at guess = {guesses[absolute_error.argmin()]:.1f}  (the median)")
print("So the loss you train with decides what your model learns to output.")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, err, name, best, best_name in [
    (axes[0], squared_error, "average squared error", y_bin.mean(), "the mean"),
    (axes[1], absolute_error, "average absolute error", np.median(y_bin), "the median"),
]:
    ax.plot(guesses, err, color="tab:blue", lw=2)
    ax.axvline(best, color="tab:red", ls="--")
    ax.text(best + 1, err.min() + (err.max() - err.min()) * 0.85, f"smallest at {best:.1f}\n= {best_name} of Y", color="tab:red")
    ax.set_xlabel("your single guess for Y"); ax.set_ylabel(name); ax.grid(alpha=0.3)
    ax.set_title(f"{name} for every possible guess")
plt.suptitle("Why E[Y | X] is 'best': the average minimises squared error (the median minimises absolute error)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "d2_why_the_average.png", dpi=120)
plt.close()


# =====================================================================
# Figure D3: with yes or no labels, the best guess is a probability
# =====================================================================
# Y is now the label: 1 for pneumonia, 0 for normal. The average of a bunch of 0s and 1s is the
# fraction of 1s. So E[Y | X] = P(pneumonia | X). A classifier that outputs a probability is
# estimating a conditional expectation.
print()
print("=== Figure D3 ===")
brightness = images.mean(axis=(1, 2))                    # X: average brightness of the whole X-ray
centres, fraction, _, counts, _ = binned_average(brightness, labels.astype(float), n_bins=12)
error_bar = np.sqrt(fraction * (1 - fraction) / counts)  # rough uncertainty of each fraction
overall = labels.mean()
print(f"Overall share of pneumonia: {overall:.3f}")
print(f"Weighted average of the per-bin shares: {np.sum(counts * fraction) / counts.sum():.3f}  (the same: averaging the averages gives the overall average)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
ax = axes[0]
ax.hist(brightness[labels == 0], bins=40, alpha=0.6, color="tab:green", density=True, label="normal")
ax.hist(brightness[labels == 1], bins=40, alpha=0.6, color="tab:red", density=True, label="pneumonia")
ax.set_xlabel("X: average brightness of the whole X-ray"); ax.set_ylabel("how common")
ax.set_title("pneumonia X-rays tend to be brighter (fluid in the lungs blocks X-rays)"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.errorbar(centres, fraction, yerr=2 * error_bar, fmt="o-", color="tab:red", capsize=3, lw=2, label="E[Y | X] = share with pneumonia in each bin")
ax.axhline(overall, color="gray", ls="--", label=f"overall share {overall:.2f}")
ax.set_ylim(0, 1.05)
ax.set_xlabel("X: average brightness of the whole X-ray"); ax.set_ylabel("P(pneumonia | X)")
ax.set_title("the best guess is now a probability"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.suptitle("Conditional expectation of a yes/no label is a probability. That is what classifiers learn.", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "d3_probability_curve.png", dpi=120)
plt.close()

print()
print("Done. Pictures are in:", OUT_DIR)
