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

# Function to load images and assign labels
def load_images(image_dir, label_index, num_classes):
    images = []
    labels = []
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, img_name)
            img = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = preprocess_input(img, version=2)
            images.append(img)
            labels.append(np.eye(num_classes)[label_index])
    return np.array(images), np.array(labels)

# Function to apply Haar cascade on a set of images
def crop_faces_in_images(images, face_cascade_path):
    cropped_images = []
    detected_indices = []
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    for idx, img in enumerate(images):
        img_uint8 = img.astype('uint8')
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cropped_face = img_uint8[y:y+h, x:x+w]
            cropped_images.append(cv2.resize(cropped_face, (224, 224)))  # Resize cropped face
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

# Define the paths and labels
haar_cascade_path = "haarcascade/haarcascade_frontalface_default.xml"
class_labels = ['ahmed', 'efaz', 'steph', 'jokic', 'timmy']
num_classes = len(class_labels)

# Load all images and labels
images, labels = [], []
for i, class_label in enumerate(class_labels):
    class_images, class_labels = load_images(f'data/{class_label}_augmented', i, num_classes)
    images.append(class_images)
    labels.append(class_labels)

# Combine all class images and labels
images = np.vstack(images)
labels = np.vstack(labels)

# Crop faces in the images
cropped_images, detected_indices = crop_faces_in_images(images, haar_cascade_path)
cropped_labels = labels[detected_indices]

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(cropped_images, cropped_labels, test_size=0.2, random_state=42)

# Load the pretrained VGGFace2 model
base_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
x = base_model.output
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_val, y_val))

# Prepare and predict test image
test_image_path = 'test_stephencurry.jpg'
test_image = image.load_img(test_image_path, target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = preprocess_input(test_image, version=2)
cropped_test_image, _ = crop_faces_in_images(test_image, haar_cascade_path)

# Display prediction result
if len(cropped_test_image) > 0:
    prediction = model.predict(cropped_test_image)
    display_image_with_label(cropped_test_image[0], prediction, class_labels)
else:
    print("No face detected in the test image.")
