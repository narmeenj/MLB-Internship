import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- FROM YOUR IMAGE ---
import tensorflow_datasets as tfds

dataset, info = tfds.load(
    "cats_vs_dogs",
    with_info=True,
    as_supervised=True
)
# ----------------------

print("==> Step 1: Splitting the Single Dataset Object Manually")
# Get total number of examples from info
DATASET_SIZE = info.splits['train'].num_examples
train_size = int(0.8 * DATASET_SIZE)
val_size = DATASET_SIZE - train_size

# Split using take and skip
train_raw = dataset.take(train_size)
val_raw = dataset.skip(train_size)

IMG_SIZE = 160
BATCH_SIZE = 32

# Preprocessing & Resize function
def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

# Optimize Data Pipelines
train_dataset = train_raw.map(preprocess_image).shuffle(1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_raw.map(preprocess_image).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

print(f"Dataset successfully split! Training examples: {train_size}, Validation examples: {val_size}")

print("==> Step 2: Creating Data Augmentation & Model Layers")
# Data Augmentation layer to prevent overfitting and boost accuracy past 93%
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.15),
])

# Base Model Setup
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze backbone

# Model Assembly
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x) # Scale pixels to [-1, 1]
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x) # Regularization
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("==> Step 3: Training the Model (Feature Extraction)")
EPOCHS = 5
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset
)

print("==> Step 4: Generating and Saving Metrics Plot")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Curves')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss Curves')
plt.savefig('learning_curves.png')
print("Saved performance graph as 'learning_curves.png'")

print("==> Step 5: Visualizing and Saving Sample Predictions")
# Take 1 batch from validation data
image_batch, label_batch = next(iter(val_dataset))
predictions = model.predict(image_batch)
predictions = tf.where(predictions < 0.5, 0, 1).numpy().flatten()

class_names = ['Cat', 'Dog']

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    pred_label = class_names[predictions[i]]
    true_label = class_names[label_batch[i]]
    
    title_color = 'green' if pred_label == true_label else 'red'
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color=title_color)
    plt.axis("off")
    
plt.tight_layout()
plt.savefig('sample_predictions.png')
print("Saved prediction grid as 'sample_predictions.png'")
