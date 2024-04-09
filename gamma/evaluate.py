import os
import cv2
import numpy as np
from keras.models import load_model

from keras.preprocessing.image import img_to_array
from keras_vggface.utils import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import json

# Load OpenCV's deep learning face detector
protoPath = "opencv/deploy.prototxt.txt"
modelPath = "opencv/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

def crop_face_from_image(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            return cv2.resize(face, (224, 224))
    return None

def load_and_preprocess_data(data_dir, class_labels):
    X, y = [], []
    for index, class_label in class_labels.items():
        class_dir = os.path.join(data_dir, class_label)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            face = crop_face_from_image(img)
            if face is not None:
                face = img_to_array(face)
                face = preprocess_input(face, version=2)
                X.append(face)
                y.append(index)
    if not X:
        return None, None
    return np.array(X), np.array(y)

# Load the model
model_path = "saved_models/models/vggface1.h5"
model = load_model(model_path)

with open('saved_models/labels/labels1.json', 'r') as f:
    class_labels = json.load(f)

# Load and preprocess the evaluation dataset
data_dir = "data_aug"
# Load and preprocess data
X_eval, y_eval = load_and_preprocess_data(data_dir, class_labels)
if X_eval is None:
    print("No data to evaluate.")
else:
    # Make predictions
    y_pred = model.predict(X_eval)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Evaluation metrics
    print("Classification Report:")
    y_eval_str = [class_labels[i] for i in y_eval]  # Convert true labels to string names
    y_pred_classes_str = [class_labels[i] for i in y_pred_classes]  # Convert predicted labels to string names

    # Now, use these string labels in classification_report
    print(classification_report(y_eval_str, y_pred_classes_str))
    print("Confusion Matrix:")
    print(confusion_matrix(y_eval, y_pred_classes))

    # Calculate and print precision, recall, F1 score, and accuracy
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_eval, y_pred_classes, average='weighted')
    accuracy = accuracy_score(y_eval, y_pred_classes)
    print(f"Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1_score:.2f}\nAccuracy: {accuracy:.2f}")

