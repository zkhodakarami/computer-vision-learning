# Journal club guide: Ambient Diffusion Posterior Sampling

A step-by-step path to understand one paper well enough to present it, answer questions about it,
and reuse its ideas in your own work.

| | |
|---|---|
| Paper | Ambient Diffusion Posterior Sampling: Solving Inverse Problems with Diffusion Models trained on Corrupted Data |
| Authors | Asad Aali and Giannis Daras (equal first authors), Brett Levac, Sidharth Kumar, Alexandros G. Dimakis, Jonathan I. Tamir |
| Where | ICLR 2025 (a top machine learning conference) |
| Paper link | https://arxiv.org/abs/2403.08728 (OpenReview page: https://openreview.net/forum?id=qeXcMutEZY) |
| Code and trained models | https://github.com/utcsilab/ambient-diffusion-mri |
| Length | 10 pages plus an 18 page appendix. You only need appendix A, B and D. |

---

## The paper in one paragraph

To speed up an MRI scan, the scanner skips most of its measurements, and a computer fills in the gaps.
Today's best gap fillers are diffusion models, which first learn what real scans look like. The catch is
that learning this normally needs thousands of complete, never-skipped scans, which are slow, costly, and
sometimes impossible to collect. This paper trains the diffusion model using only skipped, incomplete
scans, then uses it to fill in gaps for new patients. The surprise: when the new scan skips a lot
(8 times faster), the model that never saw a complete scan does better than the one trained on complete scans.

## The whole idea in one picture

```
 TRAINING (only incomplete scans exist)

   incomplete scan  --->  hide even more of it  --->  network guesses the full image
         |                                                     |
         +------- check the guess ONLY on the parts we really measured <---+

 RECONSTRUCTION (a new patient, maybe skipped in a different way)

   start from pure noise
   repeat a few hundred times:
       1. hide part of the current image the way training did, ask the network for its best guess
       2. step toward that guess                         (looks like real anatomy)
       3. nudge so the guess agrees with what the scanner measured   (matches the data)
   result: a full image
```

Step 1 of reconstruction is the paper's main new idea. Everything else is borrowed from two earlier papers,
and this guide teaches those first.

---

## How to use this guide

Do the steps in order. Each one has a goal, the pages to read, a plain explanation, and a short self check.
Do not move on until you can answer the self check without looking.

| Step | What you learn | Rough time |
|---|---|---|
| 1 | First pass: what the paper claims | 30 minutes |
| 2 | The problem: measuring is not seeing (MRI basics) | 1 to 2 hours |
| 3 | Diffusion models as a "best guess" machine | 2 hours |
| 4 | DPS: using a diffusion model to fill in missing measurements | 1 to 2 hours |
| 5 | Ambient Diffusion: learning from incomplete data | 1 to 2 hours |
| 6 | Contribution 1: Ambient Diffusion for MRI, and its proof | 2 hours |
| 7 | Contribution 2: Ambient DPS | 1 hour |
| 8 | The experiments and what they really show | 1 to 2 hours |
| 9 | Read it like a reviewer | 1 hour |
| 10 | Walk through the code | 1 to 2 hours |
| 11 | Build the talk | 2 to 3 hours |

Your lesson 3 math is the foundation here. Three parts of it come back directly:

- [Part A, the SVD with range and null space](../../03_math_you_will_lean_on/README.md#part-a-the-svd-range-and-null-space): why missing measurements cannot be recovered from the data alone.
- [Part B, Gaussian conditioning](../../03_math_you_will_lean_on/README.md#part-b-the-multivariate-gaussian-and-conditioning): filling in a hidden part from a seen part is exactly the job here.
- [Part D, conditional expectation](../../03_math_you_will_lean_on/README.md#part-d-conditional-expectation): the network in every diffusion model is learning E[clean | noisy]. This is the single most important idea in the paper.

---

## Step 1: First pass, claims only

**Goal:** know what the paper says it does before you know how.

**Read:** the abstract, the list of contributions at the bottom of page 2, the conclusion on page 10,
and look at Figures 1, 2 and 3 without reading the math.

**What to notice:**

- Figure 1 is the whole paper. Training sees one kind of damage (here, pixels erased). Reconstruction handles a different kind of damage (here, low resolution).
- Figure 2 shows images the models invent from scratch. Columns 1 to 4 were trained on incomplete scans. They look like real brains even when training skipped 7 of every 8 lines (R = 8). The models trained on the older workaround (columns 5 to 8) carry leftover stripe and ghost artifacts, which the paper calls aliasing.
- Figure 3 is the main result. At R = 4 the clean-data model (last column) wins. At R = 8 it shows heavy artifacts while the incomplete-data models stay clean.

**Self check:**
1. In one sentence, what data does the model train on?
2. In which situation does the paper claim its model beats a model trained on clean data?

---

## Step 2: The problem, measuring is not seeing

**Goal:** understand why reconstruction is hard and what "incomplete MRI data" actually means.

**Read:** paper Section 2.1, the "Multi-coil Magnetic Resonance Imaging" part (pages 4 and 5).
For background, the introduction of the fastMRI dataset paper: https://arxiv.org/abs/1811.08839

### The general shape of the problem

Almost every imaging device is a machine that turns the true image x into measurements y:

```
y = A x
```

A is called the **forward operator**: the recipe for how the device measures. When A is a short, wide matrix
(fewer measurements than pixels), it has a **null space**: directions in the image that leave no trace in y.
You met this in lesson 3, Part A, with the blur example. Any part of the image lying in the null space is
invisible to the scanner. Two very different images can give the exact same y. So the data alone cannot pick
the answer. You need extra knowledge of what real images look like. That extra knowledge is called a **prior**,
and in this paper the prior is a diffusion model.

### What an MRI scanner actually measures

| Word | Plain meaning |
|---|---|
| k-space | The MRI scanner does not take a photo. It measures the image's Fourier transform: how much of each stripe pattern (each frequency) the image contains. That collection of measurements is called k-space. |
| Phase encode line | The scanner collects k-space one horizontal or vertical line at a time. Each line costs time. |
| Undersampling | Skipping some lines to save time. |
| Acceleration factor R | Total lines divided by measured lines. R = 4 means only 1 in 4 lines is measured, so the scan is about 4 times faster. |
| Autocalibration region | The central lines of k-space (the slow, broad patterns that hold most of the image's brightness). They are always measured. In this paper, the central 20 lines. |
| Coils | The scanner listens with many small antennas at once. Each coil "sees" the body brighter near itself. |
| Sensitivity map S_i | A picture of how bright coil i sees each spot. Estimated from the central lines with a method called ESPIRiT. |
| Mask P | A list of ones and zeros saying which k-space lines were measured. |
| Adjoint | Running the measuring machine "backwards" in the simplest way: put the measured lines into empty k-space, fill the rest with zeros, transform back to an image, and combine the coils. The result is a blurry, striped first guess. |

Equation 2.5 writes what coil i measures:

```
z_i = P F S_i x + noise
```

Read it right to left: take the true image x, weight it by coil i's sensitivity S_i, take the Fourier transform F,
keep only the measured lines P. All of these steps are linear (they are matrices), so the whole thing is one
matrix A, and we are back to y = A x.

**Why the paper cares:** in some scans, such as 3D movies of contrast dye flowing through the abdomen, the body
changes faster than a full scan can be taken. Complete training scans simply cannot exist there.
(Page 4, citing Zhang et al. 2015.)

**Self check:**
1. If R = 8 and there are 320 lines, about how many are measured?
2. Why can't you recover the skipped lines from the measured ones using only math, with no prior?
3. What does the adjoint produce, and why does it look striped?

---

## Step 3: Diffusion models as a "best guess" machine

**Goal:** understand diffusion models through one idea you already know: conditional expectation.

**Read:** paper Section 2.1, first paragraph and Equation 2.1 (page 3).
Background (pick one, read the first half):
- Yang Song's blog post on score-based models: https://yang-song.net/blog/2021/score/
- Lilian Weng's blog post on diffusion models: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

The code for this paper is built on the EDM design. Skim its first pages once you are comfortable:
Karras et al. 2022, https://arxiv.org/abs/2206.00364

### The idea in plain words

Take a clean image x_0 and add Gaussian noise of strength sigma_t:

```
x_t = x_0 + sigma_t * eta,      eta = pure random noise
```

Train a network to look at the noisy image x_t and guess the clean one. Train it with squared error.
From lesson 3, Part D, you know what a network trained with squared error learns: the conditional expectation.

```
network(x_t)  ~  E[x_0 | x_t]      "the average clean image among all clean images that could have produced this noisy one"
```

That is all a diffusion model learns. It is a denoiser, at every noise level from "barely noisy" to
"pure static".

### How a denoiser becomes an image generator

Start from pure static. Ask the denoiser for its best guess. Its guess at very high noise is a blurry average
face, or average brain. Take a small step toward that guess, then add back a little fresh noise at a slightly
lower level. Repeat a few hundred times. Each round, the guess becomes sharper and more specific, and you end up
with a brand new realistic image. Equation 2.1 is exactly this loop written as a continuous process:

```
dx = -2 * (rate of noise change) * ( E[x_0 | x_t] - x_t ) dt  +  (fresh noise)
          \______________________/  \__________________/
            how big a step            "step toward the best guess"
```

The quantity `(E[x_0 | x_t] - x_t) / sigma_t^2` has a name: the **score**. It points from the noisy image toward
more likely images. The link between the denoiser and the score is called **Tweedie's formula**. You do not need
to prove it. Just remember: **the denoiser and the score carry the same information.**

**Self check:**
1. What exactly does the network output, in words?
2. Why does the network's guess look blurry at high noise? (Hint: lesson 3, Part D, the average of many possibilities.)
3. What does Equation 2.1 do in each step, in one sentence?

---

## Step 4: DPS, using a diffusion model to fill in missing measurements

**Goal:** understand how a diffusion model that only knows "what brains look like" can reconstruct one specific patient's scan.

**Read:** paper Equations 2.2 and 2.3 (page 3).
Original paper: Chung et al., "Diffusion Posterior Sampling for General Noisy Inverse Problems", ICLR 2023,
https://arxiv.org/abs/2209.14687 (read the introduction and Section 3).

### The idea in plain words

We do not want any realistic brain. We want a realistic brain **that agrees with this patient's measurements y**.
In probability language, we want samples from the **posterior** p(x | y). Bayes' rule splits it into two forces:

```
log p(x | y)  =  log p(x)        +  log p(y | x)       + constant
                 "looks real"       "matches the data"
```

So the generation loop from Step 3 gets one extra push. That is Equation 2.2. The "looks real" push comes from
the denoiser. The "matches the data" push needs p(y | x_t): how well the **noisy** in-progress image explains the
measurements. Nobody can compute that exactly.

**The DPS shortcut:** pretend the clean image is exactly the denoiser's current best guess, then measure the mismatch.

```
x0_hat   = network(x_t)                    the current best guess of the clean image
mismatch = || y - A x0_hat ||^2            how badly that guess disagrees with the scanner
push     = - gradient of mismatch with respect to x_t
```

The gradient goes backward through the network (PyTorch does this for you with autograd).
Gamma_t is a step size that sets how hard to push. In this paper it is 1 / ||y - A x0_hat||, so the push gets
stronger as the guess gets close to the data.

### DPS as a loop

```
x = pure noise
for each noise level, from high to low:
    x0_hat   = network(x)                      # best guess, the "looks real" part
    mismatch = || y - A x0_hat ||^2            # the "matches the data" part
    x = step toward x0_hat, add a little fresh noise
    x = x - gamma * gradient(mismatch, x)
return x
```

**The key property for this paper:** the network is trained once, without ever seeing A. The same network can
solve R = 2, R = 8, or even a totally different problem like deblurring. Only the mismatch line changes.

**Self check:**
1. Which line of the loop knows about the scanner, and which line knows about anatomy?
2. What is the DPS approximation, in one sentence?
3. Why can one trained network handle many acceleration factors?

---

## Step 5: Ambient Diffusion, learning from incomplete data

**Goal:** understand the training trick this paper builds on.

**Read:** paper Section 2.1, the "Ambient Diffusion" paragraph and Equation 2.4 (page 4).
Original paper: Daras et al., "Ambient Diffusion: Learning Clean Distributions from Corrupted Data", NeurIPS 2023,
https://arxiv.org/abs/2305.19256 (introduction and Section 3).

### The problem

Step 3 trained the denoiser by comparing its guess to the clean image. Without clean images, there is nothing to
compare to. If you train the network to reproduce the incomplete image, it learns to produce incomplete images,
holes and all.

### The trick, told as a game

Imagine you only have photos with some pixels erased by nature.

1. Take one of them. **Erase a few more pixels yourself.** Now the photo has two kinds of holes: nature's and yours.
2. Add diffusion noise and give this doubly damaged photo to the network. Ask for the full clean photo.
3. **Score the answer only on the pixels you actually have** (the ones nature did not erase). That includes the pixels you just erased yourself, so you know the right answer there.

The network cannot tell which holes were made by you and which by nature. They look the same. So to score well on
your holes, it must learn to fill every hole well. It ends up learning to fill holes it was never scored on.

Equation 2.4 writes this game:

```
loss = || A_train * h(A_tilde (x_0 + sigma_t eta), A_tilde)  -  y_t,train ||^2
          \_____/    \___________________________________/       \_______/
       score only on   network's guess from the doubly        the data we
       what we have    damaged input, also told the mask       really have
```

- A_train: nature's damage (the original measurement).
- A_tilde: nature's damage plus your extra damage.
- h: the network. It also receives the mask A_tilde so it knows where the holes are.

### Why it works, and the one condition

The paper states that this loss is minimized by the true best guess E[x_0 | A_tilde x_t, A_tilde], the same thing a
normal diffusion model learns. The condition is that **every pixel has some chance of being scored**.
In math: the average of A_train^T A_train, given A_tilde, must be **full rank**.

Connect it to lesson 3, Part A: "full rank" means the average scoring machine has **no null space**. If some pixel
were always missing in every training image, it would sit in the null space, the loss would never mention it,
and the network could put anything there. Random holes make sure every pixel gets checked sometimes.

**A relative you may meet in questions:** Noisier2Noise (Moran et al., CVPR 2020) uses the same "damage it more,
predict the less damaged version" idea for denoising photos.

**Self check:**
1. Why is extra damage needed? What would go wrong without it?
2. On which pixels is the loss computed?
3. State the full rank condition in plain words.

---

## Step 6: Contribution 1, Ambient Diffusion for MRI (Section 2.2 and Theorem 2.1)

**Goal:** see how the trick is adapted to real MRI, and follow the proof.

**Read:** Section 2.2 (pages 5 and 6), then appendix A.1 (pages 16 and 17).

### Three ways MRI differs from erased pixels

1. The holes are in **k-space**, not in the image. Erasing a k-space line affects the whole image as a stripe pattern.
2. The holes have **structure**: whole lines are measured or skipped, never single points.
3. There are **many coils**, each with its own view of the body.

### How they adapt the game

- **Extra damage:** delete a few more whole k-space lines from each coil's measured lines (Equation 2.6). In the experiments, data at R = 2 is further damaged to R = 3 (page 7). The central 20 lines are always kept.
- **Network input:** combine the coils with the adjoint to make one rough image (Equation 2.7). Add diffusion noise. Feed it to the network, together with the extra-damaged mask.
- **Loss:** push the network's guess through the measuring machine with the original mask (Equation 2.8), and compare only on the originally measured lines (Equation 2.10).

### Theorem 2.1 in plain words

If you train with this loss, the best possible network is exactly the true best guess E[x_0 | what it sees]. That is,
training on incomplete MRI teaches the same denoiser that training on complete MRI would, in the ideal case of a
perfect network and infinite data.

### The proof in four moves (appendix A.1)

You have all the tools. Follow along with page 16 open.

1. **Split any answer into "best guess plus error".** Write the network output as E[x_0 | input] + f(input). The goal is to show the loss is smallest only when f = 0. (Equation A.5.)
2. **Expand the square.** The loss breaks into three pieces: an error nobody can avoid, a piece that depends on how big f is, and a cross piece. (Equation A.6.)
3. **The cross piece is zero.** Once you know the network's input and the extra-damaged mask, also knowing the original mask tells you nothing more about the anatomy. So the average error of the best guess is zero, and the cross piece vanishes. (Equations A.7 to A.10.) This is the same "errors of the best guess are uncorrelated with anything you know" fact behind lesson 3, Part D.
4. **The size piece is f^T M f, with M = E[A^T A | P_tilde].** If M is full rank, this is positive for any nonzero f, so the loss is smallest only at f = 0. (Equations A.11 to A.13.) The rest of the proof shows M is full rank:
    - every k-space line has some chance of having been measured,
    - the Fourier transform is invertible, so it cannot create a null space,
    - the coil maps add up to the identity, so combining coils cannot create one either.

**Self check:**
1. In the experiments, what acceleration is the network's input, and which lines is the loss computed on?
2. Which proof step uses conditional expectation, and which uses full rank?
3. Why does the proof need the Fourier transform to be invertible?

---

## Step 7: Contribution 2, Ambient DPS (Sections 2.3 and 2.4)

**Goal:** see how the incomplete-data denoiser plugs into DPS.

**Read:** Section 2.3 (Equation 2.11), Section 2.4, and the footnote on page 6.

### The problem

DPS (Step 4) needs E[x_0 | x_t]: a denoiser that takes a complete noisy image. But the ambient network was only
ever trained on **damaged** noisy images. Giving it a complete one would be asking it to do something it never practiced.

### The fix

Pick one extra-damage mask A_tilde at random, **once**, before reconstruction starts. At every step, damage the
current noisy image with that mask before showing it to the network. Everything else is plain DPS.

```
x = pure noise
pick one random training-style mask A_tilde, keep it fixed
for each noise level, from high to low:
    x0_hat   = network(A_tilde x, A_tilde)     # the ONLY change from DPS
    mismatch = || y - A_inf x0_hat ||^2       # A_inf is the NEW patient's mask
    x = step toward x0_hat, add a little fresh noise
    x = x - gamma * gradient(mismatch, x)
return x
```

Notice the two different masks. A_tilde is the training-style damage. A_inf is how this patient was actually
scanned. They do not need to match. That is how a model trained at R = 2 can reconstruct a scan at R = 8.

### Two side results

- **Ambient One Step (A-OS), Section 2.4.** If the new patient's scan was damaged exactly like the training data, a single network call already gives the best guess. No loop needed. It is fast but only works in that matched case.
- **The error bound, Equation 2.12.** It bounds how wrong the DPS shortcut can be. In plain words: the shortcut is accurate when, given the noisy image, the clean image is already nearly determined. Skip this on a first read. It copies the DPS paper's argument.

**Self check:**
1. What is the only line that changes between DPS and Ambient DPS?
2. Why is A_tilde chosen once and kept fixed?
3. When would you use A-OS instead of A-DPS?

---

## Step 8: The experiments and what they really show

**Goal:** know the setup well enough to defend or question every number.

**Read:** Section 3 (pages 7 to 10), appendix B (data), appendix D (settings).

### The setup

| Item | Value |
|---|---|
| Data | fastMRI brain scans: 2,000 T2-weighted scans, 5 middle slices each, so 10,000 training slices |
| Training masks | Random line masks at R = 2, 4, 6, 8. Central 20 lines always kept |
| Extra damage | Delete more lines to reach R + 1 (for example R = 2 becomes R = 3) |
| Models | One ambient model per R (36 million weights). Clean-data EDM model at R = 1 (65 million weights) |
| Reconstruction | 500 steps of A-DPS |
| Test | 100 test slices, compared against the fully sampled image |

### The comparison methods in plain words

| Name | What it is |
|---|---|
| FS-DPS | A normal diffusion model trained on **complete** scans, used with DPS. The main rival. |
| FS-ALD | Another way to use a complete-scan diffusion model (Langevin dynamics, a different sampling loop). |
| L1-CS | Classic compressed sensing: no neural network, just "find the image with the simplest wavelet description that matches the data". |
| L1-DPS | Fill the training scans with L1-CS first, train a diffusion model on those filled images, then use DPS. The obvious workaround the paper wants to beat. |
| MoDL | A network trained end to end, with complete scans, for one specific R. A strong supervised reference. |
| SSDU, 1D-Partitioned SSDU | Self-supervised end-to-end networks: they also train without complete scans, by splitting each scan's measured lines into "input" and "check" groups. |
| A-ALD, A-OS | The paper's ambient model used with Langevin dynamics, or in one step. |

### The results that matter

**Images invented from scratch (Table 1, top, and Figure 2).** Lower FID is better.
Ambient models get worse as training data gets more incomplete, but stay usable.
At R = 6 and R = 8 they beat the L1-DPS workaround clearly (31.50 vs 51.43, and 48.15 vs 102.98).
The clean-data model is still best (10.41).

**Reconstruction quality (Table 2).** PSNR in decibels, higher is better. Selected rows:

| Method | Needs complete scans? | Test at R = 4 | Test at R = 8 |
|---|---|---|---|
| FS-DPS, trained on complete scans | yes | **38.17** | 29.46 |
| MoDL, trained for R = 4 | yes | 37.85 | 32.31 |
| MoDL, trained for R = 6 | yes | 37.65 | **33.21** |
| A-DPS, trained on R = 2 data | no | 36.82 | 33.19 |
| A-DPS, trained on R = 4 data | no | 35.67 | 33.05 |
| L1-CS, no network | no | 31.58 | 26.09 |

How to read it:

- At R = 4, the complete-scan diffusion model wins by about 1.4 dB. Complete data still helps when the problem is mild.
- At R = 8, A-DPS beats the complete-scan diffusion model by almost 4 dB. This is the headline. Figure 3 shows the artifacts behind that gap.
- At R = 8, A-DPS roughly ties the best supervised network (MoDL), which needed complete scans for training.

**Natural images (Section 3.3, Figures 4 to 6).** The same crossover appears on photos of animals and faces.
With many measurements the clean-data model wins. With few measurements the damaged-data models win.
The authors say plainly that **no known theory explains this crossover yet** (page 10). Damaged-data models also
needed fewer steps for the same quality (Figure 14 in the appendix).

**No ground truth at all (Figures 7 and 8, appendix).** They also trained on abdominal scans where complete data
does not exist. These results can only be judged by eye, since there is nothing to compare against.

**Self check:**
1. What is the fair one-sentence summary of Table 2?
2. Why is L1-DPS the natural rival for an "incomplete data" method?
3. What does the crossover mean, and has anyone explained it?

---

## Step 9: Read it like a reviewer

**Goal:** have your own opinion. A good journal club talk is not a sales pitch.

### Strengths

- **A real need.** Some scans can never be fully sampled. This gives them a principled prior.
- **A clean, general idea.** The forward operator at training and at reconstruction can differ. The method works for any linear measurement: MRI, CT, blur, missing pixels.
- **Theory backs the training step.** Theorem 2.1 says the training target is right, not a heuristic.
- **Open.** Code and all nine trained MRI models are public.
- **Honest.** The authors report where they lose (low acceleration, unconditional FID) and that the crossover is unexplained.

### Weak spots and questions to raise

1. **"Beats clean data" needs a footnote.** It beats a clean-data diffusion model with the same DPS solver at high acceleration. It does not beat everything. At R = 8 the supervised MoDL is equal.
2. **Network size differs.** Ambient models have 36 million weights. The clean EDM model has 65 million (appendix D). A smaller network could act as extra smoothing at high acceleration. The paper does not test a same-size clean model.
3. **Why does damaged training help at high R?** The paper's explanation is that the clean model "overfits" clean data. Other possible reasons: the ambient prior is blurrier and so more cautious, or the step size gamma suits it better. None is tested.
4. **Noise is ignored in the theory.** The theorem assumes noiseless measurements. Real k-space is always noisy. The authors point to follow-up work for the noisy case.
5. **Retrospective only.** All quantitative tests take complete scans and delete lines on a computer. Real fast scans have other issues (motion, timing). The truly undersampled abdominal data is judged only by eye.
6. **No reader study.** No radiologist rated the images. PSNR and SSIM can miss a small lesion that was erased or invented. Hallucination is not measured directly.
7. **Small test set and one contrast.** 100 test slices, brain T2 only. FID uses a network trained on everyday photos, on just 100 samples.
8. **Speed.** 500 network calls with gradients per slice. Fine for research, slow for a clinic.

---

## Step 10: Walk through the code

**Goal:** connect the equations to real lines, so you can answer "how exactly do they do X?".

Repository: https://github.com/utcsilab/ambient-diffusion-mri
It is built on NVIDIA's EDM code, so most files are standard diffusion plumbing. Focus on three places.

| File | What to look for | Matches |
|---|---|---|
| `training/loss.py`, class `AmbientLoss` | The `forward` method is the measuring machine (coil maps, Fourier transform, mask). The `adjoint` method runs it backward and sums over coils. In `__call__`, the noisy input is built with `hat_corruption_matrix` (the extra-damaged mask), the mask is attached as an extra input channel, and the guess is compared using `corruption_matrix` (the original mask). | Equations 2.6 to 2.10 |
| `training/training_loop.py` | The line `loss = val_loss`. That is the loss actually trained on. | Equation 2.10 |
| `solve_inverse_adps.py`, function `general_forward_SDE_ps` | "Euler step on Prior" damages the current image with the training-style mask, then calls the network. "Euler step on likelihood" computes the mismatch and its gradient with `torch.autograd.grad`. The extra-damage mask is created once, before the loop, with `create_masks`. | Equation 2.11 |

**Two naming traps** that will confuse you if nobody warns you:

- In `AmbientLoss`, the variable named `train_loss` is **not** what is trained. It is a diagnostic. The trained loss is `val_loss`, which compares on the original measured lines, as in the paper.
- In `solve_inverse_adps.py`, the variable `gamma` is EDM's extra-noise setting, **not** the paper's step size gamma_t. The paper's step size is `l_ss` divided by the size of the mismatch.

Running it needs a GPU with CUDA, the fastMRI data, and the trained models (a large download). Reading the code is
enough for the talk.

---

## Step 11: Build the talk

### A 25 minute slide plan

| # | Slide | Minutes | Content |
|---|---|---|---|
| 1 | Title | 0.5 | Paper, authors, ICLR 2025 |
| 2 | Why it matters | 2 | Faster MRI means skipping lines. Some scans can never be complete. Show Figure 1 |
| 3 | Measuring is not seeing | 2.5 | y = A x, k-space lines, the mask, R. The null space in one picture |
| 4 | Diffusion models in one idea | 3 | Add noise, learn the best guess E[x_0 given x_t], repeat to generate |
| 5 | DPS | 3 | Looks real plus matches data. The loop with two pushes |
| 6 | The missing piece | 1 | DPS needs a model trained on complete scans. What if we have none? |
| 7 | Ambient training | 3 | The "damage it more, score on what you have" game, with a k-space mask picture |
| 8 | Why it works | 2 | Theorem 2.1 in words. Every line must have a chance to be scored |
| 9 | Ambient DPS | 2 | The one changed line. Training mask and test mask can differ |
| 10 | Results | 3 | Figure 3 and the small PSNR table from Step 8 |
| 11 | Beyond MRI | 1 | The crossover on faces and animals. Abdominal scans with no ground truth |
| 12 | Critique | 1.5 | Three or four points from Step 9 |
| 13 | Discussion | 0.5 | Two open questions for the room |

Good open questions for the room:

- Where in our own work do we have only incomplete or noisy data that this could use?
- Why might a model trained on worse data be a better guide when the problem is very hard?
- What would you need to see before trusting this in a clinic?

### Questions the audience is likely to ask

| Question | Short answer |
|---|---|
| Didn't they use complete scans anyway, to make the masks? | Yes, for the experiments, so they could measure error. The method itself only ever touches the masked lines. The abdominal experiment has no complete scans at all. |
| How is this different from SSDU? | SSDU trains a separate network for each acceleration factor, and it only reconstructs. This trains a general prior that works with any scanning pattern at reconstruction time, and can also generate new images. |
| Does the model hallucinate? | Not measured directly. Figure 3 suggests fewer artifacts than the clean model at R = 8, but there was no reader study. |
| Why not just fill the training scans with compressed sensing first? | That is L1-DPS. It does much worse at high acceleration because the model learns the compressed sensing artifacts as if they were anatomy (Figure 2, Table 1). |
| How slow is it? | 500 network calls per slice, each followed by a gradient through the network. The paper does not report the time. |
| Does it work for CT? | The theory covers any linear measurement. CT was not tested in this paper. |

---

## Words you will meet

| Word | Plain meaning |
|---|---|
| Inverse problem | Working backward from measurements to the thing that produced them |
| Forward operator (A) | The recipe for how the device turns an image into measurements |
| Prior | What you believe real images look like before seeing the data |
| Likelihood | How well a candidate image explains the measurements |
| Posterior | Your belief about the image after seeing the data: prior and likelihood combined |
| Posterior sampling | Producing random images that are both realistic and consistent with the data |
| Diffusion model | A network trained to remove noise at every level, which can then generate images by removing noise step by step |
| Score | The direction that makes an image more likely. Carries the same information as the denoiser |
| Tweedie's formula | The link between the best-guess denoiser and the score |
| DPS | Diffusion Posterior Sampling: diffusion generation plus a push toward agreeing with the data |
| Ambient Diffusion | Training a diffusion model when you only have damaged data |
| Ambient score | The score learned by Ambient Diffusion, from damaged inputs |
| A-DPS | This paper's method: DPS using an Ambient Diffusion model |
| A-OS | Ambient One Step: one network call, only valid when test damage matches training damage |
| Full rank | A matrix with no null space: nothing is erased |
| Retrospective undersampling | Taking a complete scan and deleting lines on a computer to pretend it was a fast scan |
| PSNR, SSIM, NRMSE | Pixel-level similarity scores between a reconstruction and the truth |
| FID, LPIPS, DISTS | Scores that compare images using features from a pretrained network, closer to how images "look" |

---

## Check yourself before you present

<details>
<summary>1. Explain the paper to a radiologist in three sentences.</summary>

Fast MRI skips measurements, and an AI fills them in, but training that AI normally needs thousands of complete
slow scans. This paper trains it using only incomplete scans, by hiding even more of each scan and asking the AI to
recover what was hidden. When scans are skipped heavily, this AI actually reconstructs better than one trained on
complete scans.
</details>

<details>
<summary>2. Why can't you train a diffusion model on incomplete scans directly?</summary>

It would learn to produce incomplete images, holes and artifacts included, because those are its targets.
</details>

<details>
<summary>3. What is the single change between DPS and A-DPS?</summary>

Before calling the network, damage the current image with a fixed training-style mask and pass that mask too.
</details>

<details>
<summary>4. What condition does the training trick need, and why?</summary>

Every measurement location must have some chance of being in the loss. Otherwise those locations sit in a null space
and nothing trains the network about them.
</details>

<details>
<summary>5. Is the claim "beats clean-data models" fully fair?</summary>

Only partly. It beats a clean-data diffusion model with the same solver at high acceleration. It loses at low
acceleration, ties a supervised network at R = 8, and the clean model was a different size.
</details>

---

## Where the idea went next

- **Consistent Diffusion Meets Tweedie** (Daras, Dimakis, Daskalakis, 2024): the same group handles noisy training data exactly, which this paper's theory leaves out. https://arxiv.org/abs/2404.10177
- **Ambient Diffusion Omni** (Daras et al., NeurIPS 2025): uses low-quality and off-topic photos to train better general image generators. This is the idea leaving medical imaging. https://arxiv.org/abs/2506.10038
- **A gentle review of diffusion models for medical image reconstruction** (Webber et al., BJR Artificial Intelligence, 2024), good for placing this paper in the field. https://doi.org/10.1093/bjrai/ubae013
