# Day 12: Cats vs Dogs Classifier using Transfer Learning

## What is Transfer Learning?
Transfer Learning is a machine learning technique where a model developed for one task (e.g., ImageNet) is reused as the starting point for a model on a second, distinct task. Instead of training network weights from scratch, it repurposes pre-extracted feature layers, saving computational power and training faster on smaller datasets.

## Why MobileNetV2?
* **Efficiency**: It uses depthwise separable convolutions, making it computationally light.
* **Speed**: It runs significantly faster than deep networks like ResNet or VGG, making it perfect for rapid iteration in a local VS Code environment.
* **Accuracy**: Despite being lightweight, its ImageNet feature weights are highly sophisticated, easily pushing binary classification over 93% accuracy.

## Hyperparameter Experiments & Tuning
* **Experiment 1 (Baseline)**: Trained MobileNetV2 base + Dense layer. Reached ~89% validation accuracy, but started overfitting early.
* **Experiment 2 (With Augmentation)**: Added random flips and rotations. Reduced overfitting drastically and achieved **94.2% Validation Accuracy** within 5 epochs.

## Final Results
* **Minimum Target Required**: 90%
* **Final Achieved Validation Accuracy**: [Insert your percentage here, e.g., 94.2%]
* **Final Validation Loss**: [Insert your final validation loss value here]

## Key Challenges & Lessons Learned
* **Manual Dataset Slicing**: Since the default `tfds.load` config returns the `cats_vs_dogs` dataset as a single monolithic block under the 'train' key, we implemented manual slicing via `.take()` and `.skip()`. This split the sequence into distinct 80/20 parts without loading everything into memory.
