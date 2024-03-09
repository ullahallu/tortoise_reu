import os
import cv2
import numpy as np

# Load OpenCV's deep learning face detector
protoPath = "opencv/deploy.prototxt.txt"
modelPath = "opencv/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

def crop_and_display_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = img[startY:endY, startX:endX]
            cv2.imshow("Cropped Face", face)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            return  # Display the first detected face and return
    print(f"No face detected in {image_path}")

image_directory = 'data/jokic_augmented'

for filename in os.listdir(image_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_directory, filename)
        crop_and_display_face(image_path)
