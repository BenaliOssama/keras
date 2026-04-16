# Keras Neural Networks Quest

A hands-on introduction to building and training neural networks using Keras and TensorFlow.

## Overview

This quest guides you through the fundamentals of neural networks:
- Creating Sequential models
- Building Dense layers
- Designing network architectures
- Training and optimizing networks

The final project predicts breast cancer diagnoses using a neural network trained on real medical data.

## Exercises

### Exercise 0: Environment Setup
Set up Python 3.9+ virtual environment with required libraries.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Exercise 1: Sequential
Create an empty Sequential model container — the foundation for all layers.

```python
import keras
model = keras.Sequential()
print(model)
```

**Output:** `<Sequential name=sequential, built=False>`

### Exercise 2: Dense Layers
Build individual Dense layers with neurons and activation functions.

- Layer 1: 8 neurons, sigmoid activation, 5 inputs
- Layer 2: 4 neurons, sigmoid activation
- Layer 3: 1 neuron, sigmoid activation (output)

Each layer is a fully connected layer where every neuron connects to all neurons in the previous layer.

### Exercise 3: Architecture (Regression)
Assemble layers into a complete network for regression tasks.

```python
model = keras.Sequential()
model.add(Dense(8, input_shape=(5,), activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1, activation='linear'))  # linear for regression
```

**Key:** Output layer uses `activation='linear'` for continuous predictions.

### Exercise 4: Optimize and Train
Train a classification network on real breast cancer data and demonstrate the importance of data scaling.

**Part 1:** Train without scaling → Accuracy: ~37% (poor)

**Part 2:** Train with StandardScaler → Accuracy: >95% (excellent)

This shows how preprocessing dramatically improves learning.

## Key Concepts

### Sequential Model
A container that stacks layers in order: input → layer 1 → layer 2 → output. Information flows one direction with no shortcuts.

### Dense Layer
A fully connected layer where each neuron connects to every neuron in the previous layer. Ideal for tabular data.

### Activation Functions

**Sigmoid:** Squashes output to [0, 1]. Used for:
- Output layer in binary classification
- Hidden layers (though ReLU is faster)

**ReLU:** Max(0, x). Used for:
- Hidden layers (fast, prevents vanishing gradients)

**Linear:** No transformation. Used for:
- Output layer in regression problems

### Compile Step
Configure the network for training:
- **Optimizer:** adam (adaptive learning rate)
- **Loss Function:** binary_crossentropy (for yes/no problems)
- **Metrics:** accuracy (percentage correct)

### Training (fit)
- **epochs:** How many times to see all data (50)
- **batch_size:** How many samples before updating weights (300)
- **verbose:** Logging level (0 = silent)

### Data Scaling
StandardScaler transforms features to mean=0, std=1. Critical because:
- Features have different ranges (tumor size: 6-60, symmetry: 0.1-0.3)
- Optimizer works better with normalized inputs
- Without scaling: network struggles
- With scaling: network learns effectively

## Results

### Without Scaling
```
Final accuracy: 0.3726 (37%)
```
Network fails to learn. Unscaled features confuse the optimizer.

### With Scaling
```
Final accuracy: 0.9574 (95%+)
```
Network learns patterns effectively. Scaled features enable proper gradient updates.

## Dataset

Breast Cancer Wisconsin Dataset (sklearn.datasets)
- **569 samples** (patients)
- **30 features** (tumor measurements)
- **Target:** 0 (benign) or 1 (malignant)

## Installation & Running

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run individual exercises
python3 ex01.py  # Sequential model
python3 ex02.py  # Dense layers
python3 ex03.py  # Architecture
python3 ex04.py  # Optimize and train
```

## Requirements

- Python 3.9+
- numpy — numerical computing
- pandas — data manipulation
- jupyter — interactive notebooks
- keras — high-level deep learning API
- tensorflow — numerical computation backend
- scikit-learn — datasets and preprocessing
- matplotlib — visualization (optional)

## Architecture Summary

Final network for breast cancer classification:

```
Input (30 features)
    ↓
Dense (10 neurons, sigmoid)
    ↓
Dense (5 neurons, sigmoid)
    ↓
Dense (1 neuron, sigmoid)
    ↓
Output (probability: 0 = benign, 1 = malignant)
```

Training: 569 patients × 50 epochs, batch size 300, adam optimizer, binary crossentropy loss.

## Notes

- Results vary due to random weight initialization. Your accuracy may differ slightly from expected values.
- Warnings about CUDA are normal if GPU is unavailable — CPU computation works fine.
- Use `verbose=1` in `model.fit()` to see training progress per epoch.
- Scaling is critical: unscaled data achieves ~37% accuracy; scaled data achieves ~95%+.

## Learning Path

1. Understand Sequential models as layer containers
2. Learn Dense layers as fully connected neurons
3. Build architectures by stacking layers
4. Compile by choosing optimizer, loss, metrics
5. Train by calling fit() with data, epochs, batch size
6. Optimize by scaling data before training

This quest teaches the complete workflow for building, training, and evaluating neural networks.
