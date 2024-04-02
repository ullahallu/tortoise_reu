import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
MODELPATH = "saved_models/efficientnetb7_20240331-173402.h5"  # Or any other model you're using like EfficientNetB7
TEST_DATA_DIR = 'data/val'
IMG_SIZE = 600
BATCH_SIZE = 64
CLASS_NAMES = ['cellphone', 'wallet', 'watch']

# Load the latest model
model = load_model(MODELPATH)

# Preprocess images
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image /= 255.0
    return image

# Adjust get_label as needed
def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == CLASS_NAMES

# Load and preprocess the dataset
def load_dataset(data_dir):
    dataset = tf.data.Dataset.list_files(f"{data_dir}/*/*.jpg")
    dataset = dataset.map(lambda x: (preprocess_image(x), get_label(x)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset.batch(BATCH_SIZE)

test_ds = load_dataset(TEST_DATA_DIR)

# Function to extract true labels from the dataset
def extract_labels(dataset):
    all_labels = []
    for _, labels in dataset:
        all_labels.extend(labels.numpy())
    return np.argmax(all_labels, axis=1)

# Predict on the entire dataset
y_true = extract_labels(test_ds)
y_pred = np.argmax(model.predict(test_ds), axis=1)

# Metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Confusion Matrix
conf_mat = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()