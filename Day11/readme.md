# Day 11 Mini Project: Fashion MNIST Image Classifier Using CNN

## 🌟 Why CNNs Outperform ANNs for Image Data
1. **Spatial Hierarchy Preservation:** ANNs require multi-dimensional images to be completely flattened into a 1D vector. This process entirely discards the structural relationship between neighboring pixels. CNNs extract features while maintaining spatial coordinate information.
2. **Weight Sharing:** In an ANN, every input pixel connects to every hidden neuron, exploding parameter metrics. CNNs pass tiny matrices (kernels) across your entire image. The same structural weights look for a feature everywhere, saving computational footprint.
3. **Translational Invariance:** Once a CNN learns to recognize a specific visual element (like a sleeve or zipper), it can easily classify it regardless of where it appears in the frame.

---

## 🛠️ Components Purpose

### Convolution Layers
* **Function:** Acts as an automated mathematical scanner using custom filter kernels.
* **Purpose:** Slides across input arrays to capture spatial patterns. Early steps extract fine details (edges, lines), while deeper steps map full composite features (collars, soles, straps).

### Pooling Layers
* **Function:** Downsamples spatial volume (e.g., extracting max values inside a `2x2` window).
* **Purpose:** Trims down spatial dimensions to aggressively scale back operational parameter calculations while optimizing translation immunity against noisy structural variations.

---

## 📐 Model Architecture Blueprint

The model processing framework sequentially runs down the following structure:
* **Input Layer:** Tensor shapes of `(28, 28, 1)` matching grayscale bounds.
* **Conv2D Block 1:** 32 filters `(3x3)` + `BatchNormalization` + `MaxPooling2D` + `Dropout(0.25)`.
* **Conv2D Block 2:** 64 filters `(3x3)` + `BatchNormalization` + `MaxPooling2D` + `Dropout(0.25)`.
* **Bridge Component:** `Flatten` layer transitioning matrices into spatial vectors.
* **Dense Representation:** 128 Neurons utilizing `ReLU` activations + `Dropout(0.50)` for regularization.
* **Categorical Output Head:** 10 Hidden paths utilizing a `Softmax` distribution layout to yield classification predictions.

---

## 📊 Performance Analytics & Validation

* **Final Training Accuracy:** [Insert Your Training Accuracy e.g., 93.4%]
* **Final Test Accuracy:** [Insert Your Testing Accuracy e.g., 91.2%]

### Training Accuracy & Loss Plots
*(Include paths to plots created inside your workspace artifacts)*
![Learning Curves](artifacts/learning_curves.png)

### Model Confusion Matrix
![Confusion Matrix](artifacts/confusion_matrix.png)

*Observation Note:* Based on the confusion matrix observations, the model primarily confuses **Shirts** with **T-shirts** and **Coats** with **Pullovers** due to structural similarity.

---

## ⚡ Challenges Encountered & Resolutions
* **Overfitting Tendency:** Early setups showed test accuracy lagging behind training figures. 
  * *Resolution:* Structured explicit `Dropout` mechanisms and introduced `BatchNormalization` to scale validation variance safely.
* **Dimension Channel Error:** Encountered `ValueError` when passing raw inputs into sequential `Conv2D` channels.
  * *Resolution:* Explicitly adjusted the matrix configurations from a flat shape to structured 4D formats using `.reshape(-1, 28, 28, 1)`.

---

## 🎥 Walkthrough Video Reference
[Insert Link/Attachment of your screen recording here]
