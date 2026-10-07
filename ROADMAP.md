# Roadmap: from zero to computer vision job ready

This is the full learning path. Each step is one folder in this repo.
Every lesson has a short README in plain words and one Python script you can run.

The examples use medical images (chest X-rays, tissue slides, scans), but every skill here
is exactly what computer vision jobs ask for, no matter the industry.

## What computer vision jobs actually ask for

I read through typical job postings for computer vision engineer roles. They keep asking for the same things:

| What jobs ask for | Lessons that cover it |
|---|---|
| Python, NumPy, OpenCV | 1, 2 |
| PyTorch | 5 onward |
| Image classification with neural networks | 6, 7, 9 |
| Finding objects in images (boxes) | 11 |
| Marking exact regions in images (masks) | 10 |
| Using big pretrained models and adapting them | 9, 14, 15 |
| Scoring models the right way | 3, 8 |
| Cleaning and preparing image data | 2, 7, 12 |
| Making models fast and shipping them | 17 |
| Good coding habits: Git, tests, clean projects | 18 |
| Explaining results to non-experts | 16, 19 |

## Part A: Images and the basics

### Lesson 1: An image is just a grid of numbers
- What you learn: what an image really is to a computer. Rows, columns, color channels, pixel values from 0 to 255.
- Why jobs want it: everything else builds on this. You cannot debug a model if you do not know what it is looking at.
- What you build: load a chest X-ray, print its raw numbers, draw it, count how many images of each kind there are.

### Lesson 2: Changing images
- What you learn: resize, crop, flip, rotate, brighten, blur, find edges, and squash pixel values into a small range.
- Why jobs want it: these are the daily tools of the job. They also become "data augmentation" later (making many slightly different copies of each training image).
- What you build: one X-ray put through every operation, saved as a before-and-after picture.

### Lesson 3: A first model that learns from pixels
- What you learn: train a simple model (no neural network yet) to tell normal X-rays from pneumonia. Split data into train, validation and test, and why that matters.
- Why jobs want it: interviewers love asking why you need a separate test set. Also shows that simple models are a fair starting point.
- What you build: a scikit-learn model with an honest accuracy score.

## Part B: Neural networks

### Lesson 4: A tiny neural network from scratch
- What you learn: what a "neuron" is (multiply, add, squash), how a network guesses, measures its error, and adjusts. Built with only NumPy.
- Why jobs want it: interviewers ask you to explain training in plain words. Building it once makes that easy.
- What you build: a small network that learns on the pneumonia images.

### Lesson 5: The same thing in PyTorch
- What you learn: PyTorch does the adjusting math for you. Tensors (just grids of numbers that PyTorch can work with), the training loop, saving a model.
- Why jobs want it: PyTorch is the most requested tool in computer vision postings.
- What you build: the Lesson 4 network rewritten in PyTorch, with a progress bar.

### Lesson 6: Convolutional networks
- What you learn: why sliding a small filter over the image works better than looking at flat pixels. Layers, pooling, and how a CNN "sees".
- Why jobs want it: CNNs are still the workhorse of image work.
- What you build: a small CNN that beats Lesson 3 and Lesson 5 on the same data.

### Lesson 7: Training well
- What you learn: overfitting (memorizing instead of learning), data augmentation, learning rate, stopping early, saving the best version.
- Why jobs want it: anyone can run a training script. Jobs want people who can make it actually work.
- What you build: the Lesson 6 CNN trained properly, with plots of training progress.

### Lesson 8: Scoring medical models honestly
- What you learn: accuracy is not enough when one class is rare. Confusion matrix, sensitivity, specificity, precision, recall, ROC curve, AUC.
- Why jobs want it: picking the right score is a classic interview question, and in medicine it decides whether a model is safe.
- What you build: a full report card for the Lesson 7 model.

## Part C: The standard computer vision tasks

### Lesson 9: Borrowing a pretrained model
- What you learn: take a network already trained on millions of everyday photos and adapt it to medical images. This is called transfer learning.
- Why jobs want it: this is how almost all real projects start. Nobody trains from scratch anymore.
- What you build: a pretrained ResNet fine-tuned on tissue slide images.

### Lesson 10: Marking regions (segmentation)
- What you learn: predict a label for every pixel. The U-Net design. The Dice score for comparing two shapes.
- Why jobs want it: segmentation is a top request in medical imaging and self-driving alike.
- What you build: a U-Net that outlines a polyp in colonoscopy images.

### Lesson 11: Finding objects with boxes (detection)
- What you learn: how a model draws boxes around things. IoU (how much two boxes overlap), non-max suppression (removing duplicate boxes), and how YOLO works in plain words.
- Why jobs want it: detection is everywhere: retail, factories, security, medicine.
- What you build: run a pretrained detector, write IoU and duplicate-removal by hand.

### Lesson 12: Real medical file formats
- What you learn: DICOM (hospital scans) and NIfTI (3D brain scans). Reading them, slicing a 3D volume, CT windowing (picking a brightness range to show bone or soft tissue).
- Why jobs want it: medical imaging roles expect this. General roles value knowing how to handle odd formats.
- What you build: open a CT or MRI volume and show three slices through it.

### Lesson 13: Working in 3D
- What you learn: 3D convolutions, cutting a big volume into patches, and why memory becomes the problem.
- Why jobs want it: shows you can go beyond flat pictures.
- What you build: a small 3D classifier on organ volumes.

## Part D: Modern tools and shipping

### Lesson 14: Vision Transformers
- What you learn: the other main design besides CNNs. Cutting an image into patches and letting every patch "look at" every other patch.
- Why jobs want it: most new models are transformers. Postings now list it by name.
- What you build: use a pretrained vision transformer and compare it with the ResNet from Lesson 9.

### Lesson 15: Foundation models
- What you learn: huge models trained once that work on many tasks: CLIP (matches text and images), DINOv2 (good general image features), SAM (segments anything you click on).
- Why jobs want it: these are the current state of the art and often the fastest way to a working product.
- What you build: use SAM on a medical image, and use DINOv2 features with a simple classifier.

### Lesson 16: Showing where the model looked
- What you learn: Grad-CAM heatmaps that highlight the part of the image that drove the decision.
- Why jobs want it: doctors and customers want to trust the model. Also catches models that cheat (looking at a hospital tag instead of the lung).
- What you build: heatmaps for the Lesson 7 model.

### Lesson 17: Making it fast and shippable
- What you learn: save and load models, export to ONNX (a format that runs anywhere), measure speed, make a tiny web service that returns a prediction, Docker basics.
- Why jobs want it: "deploy" and "production" appear in most postings. Few beginners can do it.
- What you build: a local web service that takes an X-ray and returns "normal" or "pneumonia".

### Lesson 18: Good habits
- What you learn: a clean project layout, config files instead of magic numbers, fixing random seeds so results repeat, simple tests, logging experiments.
- Why jobs want it: this is the difference between a student notebook and engineer code.
- What you build: refactor one earlier lesson into a clean mini project.

### Lesson 19: Portfolio project
- What you learn: put it all together end to end.
- Why jobs want it: a finished project with a clear README is the single best thing on a junior resume.
- What you build: a pneumonia detector with data prep, training, honest scoring, heatmaps, and a small demo. Written up so a recruiter understands it in two minutes.

## Part E: Interview prep

A folder of common computer vision interview questions with short plain-language answers,
plus coding exercises interviewers actually ask (write convolution by hand, write IoU, write duplicate-box removal,
explain overfitting to a non-technical person).

## How to use this roadmap

Go in order. Each lesson takes one or two sittings. Do not move on until you can explain the lesson to a friend
in plain words. Push to GitHub after every lesson, even small ones.
