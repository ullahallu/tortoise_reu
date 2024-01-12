# RUN IN ANACONDA3 ENV "reu002"

# import libraries
import os
import cv2
import numpy as np
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# function to load images and assign labels 
def load_images(image_dir, label):
    images = []
    labels = []
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            img_path = os.path.join(image_dir, img_name)
            img = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = preprocess_input(img, version=2)  # Use version 2 for ResNet50
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# function to apply haar cascade onto a set of images, returns cropped images of haar detected faces
def crop_faces_in_images(images, face_cascade_path):
    cropped_images = []
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    for img in images:
        if img.dtype != np.uint8:
            continue  # Skip images that are not of uint8 type
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            cropped_images.append(cropped_face)
            break  # Assuming only one face per image

    return cropped_images

# Function to display the image with label
def display_image_with_label(image, label):
    window_name = "Prediction"
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0)  # Blue color in BGR
    font_scale = 1
    thickness = 2
    cv2.putText(image, label, (50, 50), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


# load images of facial classes (ahmed and efaz) and the haar cascade 
haar_cascade_path = "haarcascade/haarcascade_frontalface_default.xml"
ahmed_images, ahmed_labels = load_images('data/ahmed_augmented', [1, 0])  # Label for Ahmed
efaz_images, efaz_labels = load_images('data/steph_augmented', [0, 1])  # Label for steph
efaz_images, efaz_labels = load_images('data/jokic_augmented', [0, 1])  # Label for nikola
efaz_images, efaz_labels = load_images('data/timmy_augmented', [0, 1])  # Label for timmy
efaz_images, efaz_labels = load_images('data/efaz_augmented', [0, 1])  # Label for Efaz

# combine the datasets
images = np.concatenate((ahmed_images, efaz_images), axis=0)
labels = np.concatenate((ahmed_labels, efaz_labels), axis=0)

# apply haar cascade to images read from data
cropped_images = crop_faces_in_images(images, haar_cascade_path)


# Split the dataset into training and validation
X_train, X_val, y_train, y_val = train_test_split(cropped_images, labels, test_size=0.2, random_state=42)


# Load the pretrained VGGFace2 model
base_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


# Add custom top layers for fine-tuning
x = base_model.output
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)  # 2 classes: Ahmed and Efaz

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the model on your dataset
model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_val, y_val))

# Prepare the test image
test_image_path = 'test_image001.jpg'  # Replace with your test image path
test_image = image.load_img(test_image_path, target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = preprocess_input(test_image, version=2)

# applying haar cascade to test image
cropped_test_image = crop_faces_in_images(test_image, haar_cascade_path)

# Make a prediction
prediction = model.predict(cropped_test_image)

# Determine the label based on the prediction
predicted_label = "Ahmed" if prediction[0][0] > prediction[0][1] else "Efaz" 


# Display the image with the label
display_image_with_label(cropped_test_image, predicted_label)