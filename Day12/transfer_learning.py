import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import os

# Suppress unnecessary TensorFlow logs for cleaner terminal output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("--- Step 1: Loading Cats vs Dogs Dataset via TFDS ---")
# Load dataset splits (80% training, 20% validation)
(train_raw, val_raw), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True,
)

IMG_SIZE = 160
BATCH_SIZE = 32

# Preprocessing function for resizing
def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

# Optimize data pipeline with batching and prefetching
train_dataset = train_raw.map(preprocess_image).shuffle(1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_raw.map(preprocess_image).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

print(f"Data ready! Training batches: {len(train_dataset)}, Validation batches: {len(val_dataset)}\n")

print("--- Step 2: Building MobileNetV2 Transfer Learning Model ---")
# Load pre-trained MobileNetV2 base
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the pre-trained weights
base_model.trainable = False

# Build the complete architecture pipeline
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs) # Scale inputs to [-1, 1]
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='sigmoid')(x) # Binary classification head

model = models.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display final setup summary
model.summary()

print("\nReady for training! Run 'model.fit(train_dataset, epochs=3, validation_data=val_dataset)' to start.")
