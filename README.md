# Siamese Neural Networks for One-Shot Facial Recognition

This project implements a one-shot classification model for facial recognition using Siamese Neural Networks, inspired by the paper ["Siamese Neural Networks for One-shot Image Recognition"](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) by Koch et al., 2015.

## Project Overview

The primary goal of this project was to use convolutional neural networks (CNNs) to develop a model capable of recognizing whether two face images represent the same person. This was achieved using the "Labeled Faces in the Wild-a" (LFW-a) dataset.

Key features include:
- One-shot learning for facial recognition tasks.
- PyTorch implementation of a custom dataset handler, data transformations, and model architecture.
- Detailed experimentation with various configurations for optimization, weight initialization, and regularization.

## Dataset

The LFW-a dataset consists of labeled face images under unconstrained conditions. It includes:
- **Training Set:** 1980 pairs (985 positive, 995 negative)
- **Validation Set:** 220 pairs (115 positive, 105 negative)
- **Test Set:** 1000 pairs (500 positive, 500 negative)

**Preprocessing Steps:**
- Images resized to 105×105 pixels.
- Converted to grayscale.
- Balanced train/validation split (90:10 ratio).

More details about the dataset can be found [here](https://talhassner.github.io/home/projects/lfwa/index.html).

## Model Architecture

![image](https://github.com/user-attachments/assets/92d03aa5-17c7-4c57-8f76-b5150d94902a)

The implemented Siamese network consists of:
- **4 Convolutional Layers:** Each followed by ReLU activation.
- **3 Max-Pooling Layers:** To reduce spatial dimensions.
- **1 Fully Connected Layer:** Produces 4096-dimensional embeddings for similarity computation.

Additional configurations:
- **Batch Normalization** for faster convergence.
- **Dropout** (p=0.3) for regularization.
- Optimizer: Adam or SGD, depending on the experiment.
- **Loss Function:** Binary Cross-Entropy Loss.

## Results

Through experimentation, the best results were achieved with the following setup:
- Original architecture (Koch et al., 2015).
- Adam optimizer.
- Kaiming Uniform weight initialization.
- Batch normalization and dropout (p=0.3).
- Batch size: 64.

### Performance Metrics:
- Accuracy, loss plots, and confusion matrix available in the report results section.

## Experiments

Experiments included:
1. Comparing SGD vs. Adam optimizers.
2. Evaluating the impact of Batch Normalization and Dropout.
3. Testing reduced architectures to mitigate overfitting.

## Insights and Challenges

The project revealed:
- The importance of balancing regularization techniques like Batch Normalization and Dropout.
- Challenges in handling subtle variations in images (e.g., lighting, pose).

## Future Work

Possible extensions include:
- Introducing attention mechanisms for better focus on facial features.
- Using larger, more standardized datasets for improved generalization.


For detailed information about the implementation, refer to the [report]([./Siamese Neural Networks - project report.pdf](https://github.com/adimaman22/Siamese-Neural-Networks-for-One-shot-Image-Recognition/blob/main/Siamese%20Neural%20Networks%20-%20project%20report.pdf)).


