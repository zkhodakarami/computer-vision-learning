"""
Builds interactive/data.js: a few real images and numbers the interactive page needs.

Run from the project folder (only needed if you want to regenerate the file):
    python interactive/make_data.py

Why a .js file and not .json: the page is opened straight from disk (double-click index.html),
and browsers refuse to load .json files that way. A .js file loads fine.
"""

import json
from pathlib import Path

import numpy as np
from medmnist import PathMNIST, PneumoniaMNIST

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

train = PneumoniaMNIST(split="train", download=True, root=str(DATA_DIR))
train64 = PneumoniaMNIST(split="train", download=True, root=str(DATA_DIR), size=64)
tissue = PathMNIST(split="train", download=True, root=str(DATA_DIR))

labels = train.labels.flatten()
first_normal = int(np.argmax(labels == 0))

out = {}
out["xray28_pneumonia"] = train.imgs[0].astype(int).ravel().tolist()
out["xray28_normal"] = train.imgs[first_normal].astype(int).ravel().tolist()
out["tissue28"] = tissue.imgs[0].astype(int).ravel().tolist()          # 28*28*3, order r,g,b per pixel
out["tissue28_label"] = tissue.info["label"][str(int(tissue.labels[0][0]))]

xray64 = np.array(train64[3][0]).astype(float)
out["xray64"] = xray64.astype(int).ravel().tolist()
U, s, Vt = np.linalg.svd(xray64)
out["svd64"] = {
    "U": np.round(U, 4).ravel().tolist(),
    "S": np.round(s, 2).tolist(),
    "Vt": np.round(Vt, 4).ravel().tolist(),
}

# 1500 X-rays summarised by four numbers each: top-half brightness, bottom-half brightness,
# whole-image brightness, and the label (1 = pneumonia). Used by the conditional expectation playground.
pick = rng.choice(len(train), size=1500, replace=False)
imgs = train.imgs[pick].astype(float)
out["ce_points"] = [
    [round(float(im[:14].mean()), 1), round(float(im[14:].mean()), 1), round(float(im.mean()), 1), int(labels[i])]
    for im, i in zip(imgs, pick)
]

# The average X-ray and the first 6 PCA directions, for the "main directions" playground.
X = train.imgs[:3000].reshape(3000, -1).astype(float) / 255.0
mean_image = X.mean(axis=0)
_, s2, Vt2 = np.linalg.svd(X - mean_image, full_matrices=False)
out["pca"] = {
    "mean": np.round(mean_image, 4).tolist(),
    "directions": np.round(Vt2[:6], 4).tolist(),
    "stretch": np.round(s2[:6], 1).tolist(),
    "coords": np.round((X[:400] - mean_image) @ Vt2[:2].T, 3).tolist(),
    "coord_labels": labels[:400].tolist(),
}

text = "// Made by interactive/make_data.py. Real images from MedMNIST, stored as plain numbers.\nconst DATA = " + json.dumps(out, separators=(",", ":")) + ";\n"
(HERE / "data.js").write_text(text)
print("wrote", HERE / "data.js", f"({len(text) / 1024:.0f} KB)")
