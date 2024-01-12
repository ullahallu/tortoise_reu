# VAIRANT OF MASTER @ 11-22-2023-5:35AM
# run in anaconda3 env0002

import os
import cv2
import numpy as np
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Function to load images and assign labels
def load_images(image_dir, label, num_classes):
    images = []
    labels = []
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, img_name)
            img = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = preprocess_input(img, version=2)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Function to apply Haar cascade on a set of images
def crop_faces_in_images(images, face_cascade_path):
    cropped_images = []
    detected_indices = []  # Store indices of images where faces are detected
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    for idx, img in enumerate(images):
        img_uint8 = img.astype('uint8')
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cropped_face = img_uint8[y:y+h, x:x+w]
            cropped_images.append(cropped_face)
            detected_indices.append(idx)
    return np.array(cropped_images), detected_indices

# Function to display the image with label and confidence scores
def display_image_with_label(image, prediction, labels):
    window_name = "Prediction"
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0)
    font_scale = 0.5
    thickness = 1

    for i, score in enumerate(prediction[0]):
        text = f"{labels[i]}: {score:.2f}"
        cv2.putText(image, text, (10, 20 + i * 30), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load images and labels
haar_cascade_path = "haarcascade/haarcascade_frontalface_default.xml"
class_labels = ['Ahmed', 'Efaz', 'Steph', 'Jokic', 'Timmy']
num_classes = len(class_labels)
images, labels = [], []

for i, class_name in enumerate(class_labels):
    class_images, class_labels = load_images(f'data/{class_name}_augmented', np.eye(num_classes)[i], num_classes)
    images.append(class_images)
    labels.append(class_labels)

images = np.concatenate(images, axis=0)
labels = np.concatenate(labels, axis=0)

# Apply Haar cascade to images and get the indices of detected faces
cropped_images, detected_indices = crop_faces_in_images(images, haar_cascade_path)
labels = labels[detected_indices]  # Filter labels based on detected indices

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(cropped_images, labels, test_size=0.2, random_state=42)

# Load the pretrained VGGFace2 model
base_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_val, y_val))

# Prepare and predict test image
test_image_path = 'test_image001.jpg'
test_image = image.load_img(test_image_path, target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = preprocess_input(test_image, version=2)
cropped_test_image = crop_faces_in_images(test_image, haar_cascade_path)
prediction = model.predict(cropped_test_image)

# Display the image with the label and confidence scores
display_image_with_label(cropped_test_image[0], prediction, class_labels)
