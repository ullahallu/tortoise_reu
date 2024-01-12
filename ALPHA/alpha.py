# RUN IN ANACONDA3 ENV "reu0002"

import os
import numpy as np
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Function to load images and assign labels
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

# Load images of Ahmed and Efaz
ahmed_images, ahmed_labels = load_images('data/ahmed_augmented', [1, 0, 0])  # Label for Ahmed
efaz_images, efaz_labels = load_images('data/efaz_augmented', [0, 1, 0])  # Label for Efaz
steph_images, steph_labels = load_images('data/steph_augmented', [0, 0, 1]) #Label for Steph 

# Combine the datasets
images = np.concatenate((ahmed_images, efaz_images, steph_images), axis=0)
labels = np.concatenate((ahmed_labels, efaz_labels, steph_labels), axis=0)

# Split the dataset into training and validation
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Load the pretrained VGGFace2 model
base_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


# Add custom top layers for fine-tuning         
x = base_model.output         
x = Dense(1024, activation='relu')(x)          
predictions = Dense(3, activation='softmax')(x)  # 3 classes: Ahmed steph and Efaz          
          
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
test_image = image.load_img(test_image_path, target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = preprocess_input(test_image, version=2)


# Make a prediction
prediction = model.predict(test_image)

# Output the confidence levels
print(f'Confidence levels:')
print(f'Ahmed: {prediction[0][0]*100:.2f}%')
print(f'Efaz: {prediction[0][1]*100:.2f}%')

