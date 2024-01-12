import cv2
import numpy as np
from keras.preprocessing import image

# Function to apply Haar cascade on a single image to crop the face
def crop_face_from_image(image_path, face_cascade_path):
    # Load Haar cascade
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Load image
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img_uint8 = img.astype('uint8')

    # Convert to grayscale
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # If a face is detected, crop the face
    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_face = img_uint8[y:y+h, x:x+w]
        return cropped_face
    else:
        return None

# Path to the Haar cascade file
haar_cascade_path = 'haarcascade/haarcascade_frontalface_default.xml'

# Path to the test image
test_image_path = 'test_giannis.jpg'

# Crop the face from the image
cropped_face = crop_face_from_image(test_image_path, haar_cascade_path)

# If a face was detected, print out the cropped face
if cropped_face is not None:
    cropped_face_image = image.array_to_img(cropped_face)
    cropped_face_image.show()
else:
    print("No face detected in the image.")

