import numpy as np  
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

fashion_mnist=keras.datasets.fashion_mnsit
(X_train_images, y_train_labels), (X_test_images, y_test_labels)=fashion_mnist.load_data()


class_names=[
    "Top",
    "Pants",
    "Sweatshirt",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneakers",
    "Bag",
    "Boots"
]

plt.figure(figsize=(8,5))
if i range(10):
    