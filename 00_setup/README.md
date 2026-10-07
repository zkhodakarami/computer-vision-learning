# Setup

Do this once on any computer before running the lessons.

## 1. Get the code

```bash
git clone https://github.com/zkhodakarami/computer-vision-learning.git
cd computer-vision-learning
```

## 2. Make a private Python space for this project

A "virtual environment" is just a folder that holds the Python libraries for this project only.
It keeps them separate from other projects so nothing clashes.

```bash
python3 -m venv .venv
```

Turn it on (do this every time you open a new terminal):

```bash
source .venv/bin/activate
```

On Windows the command is `.venv\Scripts\activate` instead.

## 3. Install the libraries

```bash
pip install -r requirements.txt
```

This downloads PyTorch, OpenCV and a few others. It can take a few minutes.

## 4. If you see a certificate error (Mac only)

If a lesson stops with `CERTIFICATE_VERIFY_FAILED`, Python cannot verify websites yet.
This happens with the Python from python.org on a Mac. Run this once:

```bash
python 00_setup/fix_certificates.py
```

The other way to fix it for good is to open Finder, go to Applications, then the Python folder,
and double-click `Install Certificates.command`. That one asks for your Mac password.

## 5. Check it worked

```bash
python -c "import torch, cv2, medmnist; print('all good')"
```

If you see `all good`, you are ready for Lesson 1.

## What each library is for

| Library | What it does, in plain words |
|---|---|
| numpy | Works with grids of numbers. An image is one of these grids. |
| matplotlib | Draws pictures and charts so we can look at results. |
| pillow | Opens and saves normal image files like PNG and JPEG. |
| opencv-python | Classic image tools: resize, blur, find edges, and more. |
| scikit-learn | Simple machine learning models and ways to score them. |
| torch | PyTorch. The main tool for building and training neural networks. |
| torchvision | Ready-made image models and helpers that go with PyTorch. |
| medmnist | Small, free medical image datasets that download with one line of code. |
| tqdm | Shows a progress bar while something slow is running. |
| nibabel | Opens NIfTI files, a common format for 3D brain scans. |
| pydicom | Opens DICOM files, the format hospitals use for scans. |

## Using PyCharm

Open the project folder in PyCharm. Go to Settings, then Project, then Python Interpreter,
and pick the Python inside `.venv`. After that you can right-click any `lesson.py` and run it.
