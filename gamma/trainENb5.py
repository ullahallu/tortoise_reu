import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report

# Constants
#ENETMODEL = "EfficientNetB5"
IMG_SIZE = 456  # Adjust as needed
BATCH_SIZE = 285  # Optimized batch size
EPOCHS = 30
NUM_CLASSES = 3  # Based on your dataset

CURRENT_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
data_dir = 'data2'
model_save_path = os.path.join('saved_models', f"efficientnetb5_{CURRENT_TIME}.h5")

# Function to preprocess images
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image /= 255.0  # Normalize to [0,1] range

    return image

# Function to load and preprocess the dataset
def load_dataset(data_dir, subset):
    data_dir_path = os.path.join(data_dir, subset)
    dataset = tf.data.Dataset.list_files(f"{data_dir_path}/*/*.jpg")
    dataset = dataset.map(lambda x: (preprocess_image(x), get_label(x)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

CLASS_NAMES = np.array([item.name for item in os.scandir(data_dir + '/train') if item.is_dir()])

# Function to get label from file path
def get_label(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES

# Apply data augmentation
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.5)
    return image, label

# Preparing the dataset
train_ds = load_dataset(data_dir, 'train')
val_ds = load_dataset(data_dir, 'val')


train_ds = train_ds.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Model building function
def build_model(num_classes):
    base_model = EfficientNetB5(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = True

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=True)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model(NUM_CLASSES)

# Callbacks
callbacks = [
    ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max'),
    CSVLogger('training_log.csv', append=True)
]

# Training
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks
)

# Post-training evaluation and plotting code remains the same
# Plotting Training History
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.show()

# Evaluation Helper Function (Updated for Larger Batch Size)
def evaluate_model(model, validation_data, batch_size=BATCH_SIZE):
    # Adjust the batch size for evaluation if needed
    validation_data = validation_data.unbatch().batch(batch_size)
    true_labels = np.concatenate([y for x, y in validation_data], axis=0)
    predictions = model.predict(validation_data)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1)
    cm = confusion_matrix(true_classes, predicted_classes)
    cr = classification_report(true_classes, predicted_classes)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

# Call evaluate_model function if you need to evaluate
#evaluate_model(model, val_ds)