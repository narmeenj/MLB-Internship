import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix

output_dir = "Day11/artifacts"
os.makedirs(output_dir, exist_ok=True)

print("Loading and preprocessing Fashion MNIST...")
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

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

train_images = train_images / 255.0
test_images = test_images / 255.0

X_train = train_images.reshape(-1, 28, 28, 1)
X_test = test_images.reshape(-1, 28, 28, 1)

print("Generating sample image visual...")
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.suptitle("Fashion MNIST - Dataset Overview Sample", fontsize=14)
plt.savefig(f"{output_dir}/sample_dataset_images.png")
plt.show()


print("Initializing CNN Architecture...")
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


print("Starting Model Training...")
history = model.fit(X_train, train_labels, 
                    epochs=15, 
                    batch_size=64, 
                    validation_split=0.2)

print("\nEvaluating Model Performance on Test Data...")
test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=2)
print(f"\n⚡ Final Test Accuracy: {test_acc*100:.2f}%")
print(f"⚡ Final Test Loss: {test_loss:.4f}")

print("Plotting training logs...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='#ff7f0e', linewidth=2)
ax1.set_title('Training & Validation Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
ax2.plot(history.history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
ax2.set_title('Training & Validation Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.savefig(f"{output_dir}/learning_curves.png")
plt.show()

print("[INFO] Calculating Confusion Matrix...")
predictions = model.predict(X_test)
y_pred_classes = np.argmax(predictions, axis=1)

cm = confusion_matrix(test_labels, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Fashion MNIST')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.savefig(f"{output_dir}/confusion_matrix.png")
plt.show()

correct_indices = np.where(y_pred_classes == test_labels)[0]
incorrect_indices = np.where(y_pred_classes != test_labels)[0]

plt.figure(figsize=(12, 6))
for i, idx in enumerate(correct_indices[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[idx], cmap='Greens')
    plt.title(f"Pred: {class_names[y_pred_classes[idx]]}\nTrue: {class_names[test_labels[idx]]}", fontsize=9)
    plt.axis('off')
plt.suptitle("10 Correctly Classified Sample Images", fontsize=14, color='green')
plt.tight_layout()
plt.savefig(f"{output_dir}/correct_classifications.png")
plt.show()

plt.figure(figsize=(12, 6))
for i, idx in enumerate(incorrect_indices[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[idx], cmap='Reds')
    plt.title(f"Pred: {class_names[y_pred_classes[idx]]}\nTrue: {class_names[test_labels[idx]]}", fontsize=9)
    plt.axis('off')
plt.suptitle("10 Miscalibrated / Incorrectly Classified Sample Images", fontsize=14, color='red')
plt.tight_layout()
plt.savefig(f"{output_dir}/incorrect_classifications.png")
plt.show()

print(f"[SUCCESS] All tasks completed! Files generated successfully in standard directory: {output_dir}")
