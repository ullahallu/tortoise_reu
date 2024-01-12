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

# Function to load images and assign labels
def load_images(image_dir, label, face_cascade_path):
    images = []
    labels = []
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, img_name)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224))
                face = image.img_to_array(face)
                face = preprocess_input(face, version=2)
                images.append(face)
                labels.append(label)
                break  # Assuming only one face per image

    return np.array(images), np.array(labels)

# Function to crop face using Haar Cascade
def crop_face(image_path, face_cascade_path):
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        return cropped_face
    return None  # Return None if no face is detected

# Load images of Ahmed and Efaz
haar_cascade_path = "haarcascade\haarcascade_frontalface_default.xml"
ahmed_images, ahmed_labels = load_images('data/ahmed', [1, 0], haar_cascade_path)  # Label for Ahmed
efaz_images, efaz_labels = load_images('data/efaz', [0, 1], haar_cascade_path)  # Label for Efaz

# Combine the datasets
images = np.concatenate((ahmed_images, efaz_images), axis=0)
labels = np.concatenate((ahmed_labels, efaz_labels), axis=0)

# Split the dataset into training and validation
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

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
test_image_path = 'test_stephencurry.jpg'  # Replace with your test image path
cropped_test_image = crop_face(test_image_path, haar_cascade_path)

if cropped_test_image is not None:
    cropped_test_image_resized = cv2.resize(cropped_test_image, (224, 224))
    cropped_test_image_resized = image.img_to_array(cropped_test_image_resized)
    cropped_test_image_resized = np.expand_dims(cropped_test_image_resized, axis=0)
    cropped_test_image_resized = preprocess_input(cropped_test_image_resized, version=2)

    # Make a prediction on the cropped image
    prediction = model.predict(cropped_test_image_resized)

    # Determine the label based on the prediction
    predicted_label = "Ahmed" if prediction[0][0] > prediction[0][1] else "Efaz"

    # Display the cropped test image with the label
    display_image_with_label(cropped_test_image, predicted_label)
else:
    print("No face detected in the test image.")