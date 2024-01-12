from flask import Flask, request, jsonify
import sqlite3
import os
from PIL import Image
from data_augmentation import augment_images

app = Flask(__name__)

DATABASE = 'tortoise_0002.db'
UPLOAD_FOLDER = 'uploaded_images'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/upload-facial-class', methods=['POST'])
def upload_facial_class():
    face_class_name = request.form['face_class_name']
    relationship_to_user = request.form['relationship_to_user']
    user_id = request.form['user_id']

    conn = get_db_connection()
    cursor = conn.cursor()

    # Insert new facial class with placeholder image folder path
    cursor.execute('INSERT INTO facial_classes (face_class_name, relationship_to_user, user_id, image_folder_path) VALUES (?, ?, ?, ?)',
                   (face_class_name, relationship_to_user, user_id, ''))
    conn.commit()
    face_class_id = cursor.lastrowid

    # Create directory for facial class and augment images
    folder_path = os.path.join(UPLOAD_FOLDER, str(face_class_id))
    images = [Image.open(img.stream) for img in request.files.getlist('images')]
    augment_images(images, folder_path)

    # Update facial_classes with actual image folder path
    cursor.execute('UPDATE facial_classes SET image_folder_path = ? WHERE face_class_id = ?', (folder_path, face_class_id))
    
    # Insert record into user_facial_classes
    cursor.execute('INSERT INTO user_facial_classes (user_id, face_class_id, relationship_status) VALUES (?, ?, ?)',
                   (user_id, face_class_id, relationship_to_user))

    conn.commit()
    conn.close()

    return jsonify({'message': 'Facial class uploaded successfully', 'face_class_id': face_class_id})

@app.route('/object-detection-history/<int:user_id>', methods=['GET'])
def object_detection_history(user_id):
    conn = get_db_connection()
    objects = conn.execute('SELECT * FROM detected_objects WHERE user_id = ?', (user_id,)).fetchall()
    conn.close()
    return jsonify([dict(row) for row in objects])

if __name__ == '__main__':
    app.run(debug=True)
