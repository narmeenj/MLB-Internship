import numpy as np  
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

fashion_mnist=keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,test_labels)=fashion_mnist.load_data()


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
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.suptitle("Sample Fashion MNIST Images", fontsize=16)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\n--- Training Starting ---")
history = model.fit(train_images, train_labels, 
                    epochs=10, 
                    batch_size=64, 
                    validation_split=0.2)

print("\n--- Model Evaluation ---")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
print(f"Final Test Loss: {test_loss:.4f}")

predictions = model.predict(test_images)

plt.figure(figsize=(6,3))
plt.imshow(test_images[0].reshape(28,28), cmap=plt.cm.binary)
predicted_label = np.argmax(predictions[0])
true_label = test_labels[0]
plt.title(f"Predicted: {class_names[predicted_label]} | True: {class_names[true_label]}")
plt.axis('off')
plt.show()    