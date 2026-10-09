# Learn by playing: the interactive page

A single web page where every concept from the lessons becomes something you can touch:
drag a slider, click a picture, and watch the numbers change. Each playground comes with a few
small experiments to try and each lesson has a quiz. Your progress is saved in your browser.

## Open it

Double-click `interactive/index.html`. That is all. No server, no install, works offline.

If your browser complains about loading local files, run this from the project folder and
open http://localhost:8765 instead:

```bash
.venv/bin/python -m http.server 8765 --directory interactive
```

## What is inside

| Lesson | Playgrounds |
|---|---|
| 1. Images are numbers | hover over a real X-ray and read its pixel numbers; mix red, green and blue; take a colour image apart |
| 2. Changing images | brightness, contrast, blur, rotate, flip, edges and threshold on a real X-ray, with a live histogram and the normalize table |
| 3. Math you will lean on | what a 2 by 2 matrix does to a circle (SVD, rank, range, null space); rebuild an X-ray from k pieces; walk along the main directions of 3000 X-rays (PCA); slice the Gaussian hill (conditioning); zoom in on a bendy map (the Jacobian); divergence of six vector fields; the best guess curve (conditional expectation) |

## Files

| File | What it is |
|---|---|
| `index.html` | the page |
| `style.css` | how it looks, including dark mode |
| `app.js` | all the playgrounds, quizzes and navigation, in plain JavaScript with no libraries |
| `data.js` | a few real X-rays and some precomputed numbers, written by `make_data.py` |
| `make_data.py` | regenerates `data.js` from the MedMNIST data (only needed if you change what data the page uses) |

## Adding a playground for a new lesson

Open `app.js` and copy any `concept({ ... })` block. Give it a new `id`, the right `lesson`,
a plain-words `idea`, a few `tries`, and a `build(root)` function that adds controls and canvases
to `root`. Add quiz questions with `quiz('l4', [...])`. Add the lesson to the `LESSONS` list and
to the `QUIZZES` object. That is all: the sidebar, progress and paging pick it up automatically.
