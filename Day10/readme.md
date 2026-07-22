# Day 10 Mini Project: My First Artificial Neural Network (ANN)

## 🧠 Theoretical Fundamentals

### What is Deep Learning?
Deep Learning is a branch of Machine Learning inspired by the structure of the human brain. It uses layered networks to automatically discover patterns in data without humans needing to clean or extract features manually.

### Machine Learning vs. Deep Learning
* **Machine Learning**: Works best on smaller datasets. It requires humans to manually select and engineer features from the data.
* **Deep Learning**: Scales incredibly well with massive datasets. It automatically learns features using stacked layers of computational neurons.

### What is a Perceptron?
A Perceptron is the basic building block of a neural network. It takes numeric inputs, multiplies them by weight adjustments, adds an offset value called a bias, and runs the total through an activation function to generate an output.

### Activation Functions Explored
* **ReLU**: Turns negative numbers into 0 and keeps positive numbers exactly as they are. It is the default filter used in hidden layers to speed up calculation training.
* **Sigmoid**: Squeezes numbers into a true percentage probability between 0 and 1. It is used in 1-neuron output layers for simple Yes/No questions.
* **Tanh**: Compresses numbers symmetrically between -1 and 1. It is used in internal layers that require zero-centered data scaling.
* **Softmax**: Distributes percentages across multiple categories so the total sum adds up to exactly 100%. It is the standard filter used for multi-class classification tasks.

## 📊 Project Outcomes & Model Performance
* **Dataset Used**: Fashion MNIST (60,000 training images, 10,000 testing images of clothing items).
* **Network Design**: A Sequential framework flattening a 28x28 image grid (784 inputs) through a 128-neuron ReLU hidden layer down to a 10-neuron Softmax classification output layer.
* **Final Performance Metrics**:
  * **Final Training Accuracy**: ~88.9%
  * **Final Testing Accuracy**: ~87.3%
