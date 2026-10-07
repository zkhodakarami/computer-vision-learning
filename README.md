# Computer Vision Learning

My step-by-step path from the basics of images to being ready for computer vision roles.
All examples use medical images (chest X-rays, tissue slides, scans), but the skills apply everywhere.

Every lesson is one folder with:
- a short `README.md` that explains the idea in plain words, no jargon
- one `lesson.py` you can run from top to bottom, with comments on every step
- an `outputs/` folder with the pictures the script makes

## Start here

1. Follow [00_setup](00_setup/README.md) once to install everything.
2. Open [ROADMAP.md](ROADMAP.md) to see the full path and why each step matters for jobs.
3. Do the lessons in order.

To run any lesson:

```bash
source .venv/bin/activate
python 01_images_are_numbers/lesson.py
```

## Progress

| # | Lesson | Status |
|---|---|---|
| 1 | [An image is just a grid of numbers](01_images_are_numbers/README.md) | Done |
| 2 | [Changing images](02_changing_images/README.md) | Done |
| 3 | [Math you will lean on](03_math_you_will_lean_on/README.md) | Done |
| 4 | A first model that learns from pixels | Next |
| 5 | A tiny neural network from scratch | Later |
| 6 | The same thing in PyTorch | Later |
| 7 | Convolutional networks | Later |
| 8 | Training well | Later |
| 9 | Scoring medical models honestly | Later |
| 10 | Borrowing a pretrained model | Later |
| 11 | Marking regions (segmentation) | Later |
| 12 | Finding objects with boxes (detection) | Later |
| 13 | Real medical file formats | Later |
| 14 | Working in 3D | Later |
| 15 | Vision Transformers | Later |
| 16 | Foundation models | Later |
| 17 | Showing where the model looked | Later |
| 18 | Making it fast and shippable | Later |
| 19 | Good habits | Later |
| 20 | Portfolio project | Later |

## Dataset

Most lessons use [MedMNIST](https://medmnist.com/), a set of small, free medical image collections.
It downloads with one line of code, needs no login, and runs fine on a laptop.
Downloaded files go into `data/`, which is not pushed to GitHub.
