import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report


IMG_SIZE = 600  # Adjust for EfficientNetB7
BATCH_SIZE = 4
EPOCHS = 10
NUM_CLASSES = 3

data_dir = 'data2'
CURRENT_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
model_save_path = os.path.join('saved_models', f"efficientnetb7_{CURRENT_TIME}.h5")

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image /= 255.0
    return image

def load_dataset(data_dir, subset):
    data_dir_path = os.path.join(data_dir, subset)
    dataset = tf.data.Dataset.list_files(f"{data_dir_path}/*/*.jpg")
    dataset = dataset.map(lambda x: (preprocess_image(x), get_label(x)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

CLASS_NAMES = np.array([item.name for item in os.scandir(data_dir + '/train') if item.is_dir()])

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == CLASS_NAMES

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.5)
    return image, label

train_ds = load_dataset(data_dir, 'train').map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
val_ds = load_dataset(data_dir, 'val').batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

def build_model(num_classes):
    base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = True  # Fine-tuning

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=True)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model(NUM_CLASSES)

callbacks = [
    ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max'),
    CSVLogger('training_log.csv', append=True)
]

history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks)

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
