# CPU implmenetation of tuning VGGFace2 (pret. on imagenet) 
# utilizes openCVs deep learning face detector

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

# Load OpenCV's deep learning face detector
protoPath = "opencv/deploy.prototxt.txt"  # Update this path
modelPath = "opencv/res10_300x300_ssd_iter_140000.caffemodel"  # Update this path
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

def display_image_with_label(img, label):
    print(f"Displaying image with label: {label}")  # For debugging
    window_name = "Prediction"
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0)  # Blue color in BGR
    font_scale = 1
    thickness = 2
    cv2.putText(img, label, (50, 50), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.imshow(window_name, img)
    cv2.waitKey(10000)  # Display for 1000 ms
    cv2.destroyAllWindows()


def load_images(image_dir, label):
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

def crop_faces_in_images(images, labels):
    cropped_images = []
    new_labels = []
    faces_detected = 0
    for img, label in zip(images, labels):
        if img is None or img.size == 0:
            continue
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                faces_detected += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = img[startY:endY, startX:endX]
                face = cv2.resize(face, (224, 224))
                cropped_images.append(face)
                new_labels.append(label)
                break
    return np.array(cropped_images), np.array(new_labels), faces_detected

class_labels = ['giannis', 'steph', 'jokic', 'timmy', 'kobe']
num_classes = len(class_labels)
images, labels = [], []

for i, class_label in enumerate(class_labels):
    class_images, class_labels = load_images(f'data/{class_label}_augmented', np.eye(num_classes)[i])
    cropped_images, new_labels, faces_detected = crop_faces_in_images(class_images, class_labels)
    print(f"Class {class_label}: Loaded {len(class_images)} images, Detected {faces_detected} faces")
    images.extend(cropped_images)
    labels.extend(new_labels)

X_train, X_val, y_train, y_val = train_test_split(np.array(images), np.array(labels), test_size=0.2, random_state=42)

base_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
x = base_model.output
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_val, y_val))

test_image_path = 'images.jpg'
test_img = cv2.imread(test_image_path)
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
test_img = cv2.resize(test_img, (224, 224))
test_img = np.expand_dims(test_img, axis=0)
test_img = test_img.astype('float32')
test_img = preprocess_input(test_img)

cropped_test_images, _, faces_detected = crop_faces_in_images(test_img, ['test'])
if faces_detected > 0:
    prediction = model.predict(cropped_test_images)
    predicted_label_index = np.argmax(prediction)
    predicted_label_name = class_labels[predicted_label_index]  # This should already be a string.
    print(f"Predicted class: {predicted_label_name}")

    # Ensure the label is definitely a string.
    predicted_label_name_str = str(predicted_label_name)

    # Assuming your preprocessing scales pixel values to [-127.5, 127.5], revert it for display.
    # If your preprocessing differs, adjust this step accordingly.
    display_img = ((cropped_test_images[0] * 127.5) + 127.5).astype(np.uint8)

    # Ensure the image is in BGR format for OpenCV display.
    display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
    
    display_image_with_label(display_img, predicted_label_name_str)
else:
    print("No face detected in the test image.")
