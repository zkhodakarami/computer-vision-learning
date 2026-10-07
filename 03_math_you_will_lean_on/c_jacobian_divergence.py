"""
Lesson 3, part C: Jacobians and the divergence of a vector field.

Run from the project folder:
    python 03_math_you_will_lean_on/c_jacobian_divergence.py

The one-sentence ideas:
    Jacobian:   zoom in far enough on any smooth bendy map and it looks flat. The Jacobian is
                that flat map: a small matrix saying how a tiny step in each input direction
                moves the output.
    Divergence: at each point of a field of arrows, does the flow spread out (positive),
                squeeze in (negative), or neither (zero)? It is the sum of the Jacobian's diagonal.

Pictures saved into 03_math_you_will_lean_on/outputs/
    c1_jacobian_zoom.png         a bendy map, and its flat (Jacobian) approximation at one point
    c2_divergence_fields.png     source, sink, swirl, and a real warp field coloured by divergence
    c3_xray_warp.png             warp an X-ray and map where tissue was squeezed or stretched
"""

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from medmnist import PneumoniaMNIST

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"
OUT_DIR = HERE / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)
np.set_printoptions(precision=3, suppress=True)


# ---------- The bendy map we study ----------
# It takes a point (x, y) and moves it a little, by an amount that depends on where it is.
def warp(x, y):
    new_x = x + 0.3 * np.sin(1.2 * x) + 0.15 * np.sin(1.2 * y)
    new_y = y + 0.3 * np.sin(1.2 * y) + 0.15 * np.sin(1.2 * x)
    return new_x, new_y


def jacobian_by_hand(x, y):
    """The Jacobian worked out with pen and paper: d(new_x)/dx, d(new_x)/dy on row 1, same for new_y on row 2."""
    return np.array([[1 + 0.36 * np.cos(1.2 * x), 0.18 * np.cos(1.2 * y)],
                     [0.18 * np.cos(1.2 * x), 1 + 0.36 * np.cos(1.2 * y)]])


def jacobian_by_nudging(f, x, y, h=1e-5):
    """The Jacobian found numerically: nudge each input a tiny bit and see how much each output moves."""
    fx, fy = f(x, y)
    fx_dx, fy_dx = f(x + h, y)          # nudge x
    fx_dy, fy_dy = f(x, y + h)          # nudge y
    return np.array([[(fx_dx - fx) / h, (fx_dy - fx) / h],
                     [(fy_dx - fy) / h, (fy_dy - fy) / h]])


def divergence_by_hand(x, y):
    """Divergence of the displacement (how far each point moved) = sum of the Jacobian's diagonal minus 2."""
    return 0.36 * np.cos(1.2 * x) + 0.36 * np.cos(1.2 * y)


# =====================================================================
# Figure C1: the Jacobian is the flat approximation of the bendy map
# =====================================================================
print("=== Figure C1 ===")
px, py = 0.8, -0.6                                        # the point we zoom in on
J_hand = jacobian_by_hand(px, py)
J_nudge = jacobian_by_nudging(warp, px, py)

# PyTorch can also work out the Jacobian for us. This is the same machinery that trains neural networks.
def warp_torch(v):
    x, y = v[0], v[1]
    return torch.stack([x + 0.3 * torch.sin(1.2 * x) + 0.15 * torch.sin(1.2 * y),
                        y + 0.3 * torch.sin(1.2 * y) + 0.15 * torch.sin(1.2 * x)])
J_torch = torch.autograd.functional.jacobian(warp_torch, torch.tensor([px, py], dtype=torch.float64)).numpy()

print(f"Jacobian at ({px}, {py}):")
print("  by hand:\n", J_hand)
print("  by nudging:\n", J_nudge)
print("  by PyTorch autograd:\n", J_torch)
print("All three agree. det =", f"{np.linalg.det(J_hand):.3f}", "(area scale factor near this point)")

# Draw a grid before and after the warp.
lines = np.linspace(-2.5, 2.5, 21)
fine = np.linspace(-2.5, 2.5, 200)
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
for ax, warped in zip(axes[:2], [False, True]):
    for v in lines:
        for xs, ys in [(np.full_like(fine, v), fine), (fine, np.full_like(fine, v))]:
            if warped:
                xs, ys = warp(xs, ys)
            ax.plot(xs, ys, color="gray", lw=0.6)
    ax.set_aspect("equal"); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)

# A small square around the point, before and after.
half = 0.35
corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half], [-half, -half]])
edge = np.vstack([np.linspace(corners[i], corners[i + 1], 40) for i in range(4)])    # points along the square's edge
axes[0].plot(px + edge[:, 0], py + edge[:, 1], color="tab:blue", lw=2)
axes[0].plot(px, py, "o", color="tab:red")
axes[0].set_title("before: a regular grid and a small square around the point")

true_x, true_y = warp(px + edge[:, 0], py + edge[:, 1])                       # where the square's edge really goes
fpx, fpy = warp(px, py)
flat = np.array([fpx, fpy]) + edge @ J_hand.T                                 # where the Jacobian says it goes
for ax in axes[1:]:
    ax.plot(true_x, true_y, color="tab:blue", lw=2, label="true image of the square (curvy)")
    ax.plot(flat[:, 0], flat[:, 1], "--", color="tab:red", lw=2, label="Jacobian's flat guess (a parallelogram)")
    ax.plot(fpx, fpy, "o", color="tab:red")
axes[1].set_title("after: the warped grid, and both versions of the square")
axes[1].legend(fontsize=8, loc="lower right")
axes[2].set_xlim(fpx - 0.7, fpx + 0.7); axes[2].set_ylim(fpy - 0.7, fpy + 0.7); axes[2].set_aspect("equal")
axes[2].grid(alpha=0.3)
axes[2].set_title("zoomed in: the flat guess is almost exactly right")
plt.suptitle("The Jacobian: zoom in on a bendy map and it looks like a flat one (a matrix)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "c1_jacobian_zoom.png", dpi=120)
plt.close()


# =====================================================================
# Figure C2: divergence of four fields of arrows
# =====================================================================
print()
print("=== Figure C2 ===")
gx, gy = np.meshgrid(np.linspace(-2.5, 2.5, 60), np.linspace(-2.5, 2.5, 60))
step = gx[0, 1] - gx[0, 0]
fields = [
    ("source: arrows flow outward", gx, gy),
    ("sink: arrows flow inward", -gx, -gy),
    ("swirl: arrows go around", -gy, gx),
    ("our warp: how far each point moved", warp(gx, gy)[0] - gx, warp(gx, gy)[1] - gy),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 11))
for ax, (title, vx, vy) in zip(axes.flat, fields):
    # Numerical divergence: change of the x-arrow along x, plus change of the y-arrow along y.
    dvx_dx = np.gradient(vx, step, axis=1)
    dvy_dy = np.gradient(vy, step, axis=0)
    div = dvx_dx + dvy_dy
    limit = max(abs(div).max(), 0.5)            # keep a sensible colour scale even when divergence is 0 everywhere
    im = ax.pcolormesh(gx, gy, div, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
    skip = (slice(None, None, 5), slice(None, None, 5))
    ax.quiver(gx[skip], gy[skip], vx[skip], vy[skip], color="black", scale=30, width=0.004)
    ax.set_aspect("equal"); ax.set_title(f"{title}\ndivergence from {div.min():.2f} to {div.max():.2f}", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)
    print(f"{title.split(':')[0]:>8}: divergence ranges {div.min():.2f} .. {div.max():.2f}")
check = divergence_by_hand(0.8, -0.6)
i, j = np.argmin(abs(gy[:, 0] + 0.6)), np.argmin(abs(gx[0] - 0.8))
print(f"Check at (0.8, -0.6): by hand {check:.3f}, numerically {(np.gradient(fields[3][1], step, axis=1) + np.gradient(fields[3][2], step, axis=0))[i, j]:.3f}")
plt.suptitle("Divergence: red = spreading out (positive), blue = squeezing in (negative), white = neither", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "c2_divergence_fields.png", dpi=120)
plt.close()


# =====================================================================
# Figure C3: warp an X-ray and map where tissue was squeezed or stretched
# =====================================================================
# When two scans of the same patient are lined up (called registration), the result is a warp
# like ours. Doctors then look at the Jacobian determinant map: above 1 means that spot grew,
# below 1 means it shrank. Here we fake a warp on one X-ray to show the idea.
print()
print("=== Figure C3 ===")
train64 = PneumoniaMNIST(split="train", download=True, root=str(DATA_DIR), size=64)
xray = np.array(train64[3][0])
size = xray.shape[0]

# For every output pixel, ask the warp where to look in the original.
cols, rows = np.meshgrid(np.arange(size), np.arange(size))
to_math = lambda p: (p - (size - 1) / 2) / ((size - 1) / 2) * 2.5          # pixel index -> coordinate in -2.5..2.5
to_pixel = lambda c: c / 2.5 * ((size - 1) / 2) + (size - 1) / 2            # and back
src_x, src_y = warp(to_math(cols), to_math(rows))
map_x, map_y = to_pixel(src_x).astype(np.float32), to_pixel(src_y).astype(np.float32)
warped = cv2.remap(xray, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# A grid picture pushed through the same warp, so we can see the squeeze and stretch.
grid_img = np.full((size, size), 255, np.uint8)
grid_img[4::8, :] = 0; grid_img[:, 4::8] = 0
warped_grid = cv2.remap(grid_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# Jacobian determinant of the sampling map, estimated numerically like registration software does.
dxdc, dxdr = np.gradient(map_x, axis=1), np.gradient(map_x, axis=0)
dydc, dydr = np.gradient(map_y, axis=1), np.gradient(map_y, axis=0)
det = dxdc * dydr - dxdr * dydc
print(f"Jacobian determinant over the image: smallest {det.min():.2f}, largest {det.max():.2f}")
print("Above 1: this output patch gathered a bigger patch of the original, so it looks squeezed. Below 1: stretched.")

fig, axes = plt.subplots(1, 4, figsize=(17, 5.6))
axes[0].imshow(xray, cmap="gray", vmin=0, vmax=255); axes[0].set_title("original X-ray")
axes[1].imshow(warped, cmap="gray", vmin=0, vmax=255); axes[1].set_title("warped X-ray")
axes[2].imshow(warped_grid, cmap="gray", vmin=0, vmax=255); axes[2].set_title("a grid pushed through the same warp\nsmall cells = squeezed, big cells = stretched")
im = axes[3].imshow(det, cmap="RdBu_r", vmin=2 - det.max(), vmax=det.max())
axes[3].set_title("Jacobian determinant\nred above 1 (squeezed), blue below 1 (stretched)")
plt.colorbar(im, ax=axes[3], fraction=0.046)
for ax in axes:
    ax.axis("off")
plt.suptitle("What registration software reports: where did the picture get squeezed or stretched?", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(OUT_DIR / "c3_xray_warp.png", dpi=120)
plt.close()

print()
print("Done. Pictures are in:", OUT_DIR)
