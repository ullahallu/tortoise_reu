# script for running Flask server 

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename  # Add this import
import sqlite3
import os

app = Flask(__name__)

DATABASE = 'tortoise_0002.db'
UPLOAD_FOLDER = 'uploaded_images'  # Folder where images will be stored

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# endpoint for uploading new facial class
@app.route('/upload-facial-class', methods=['POST'])
def upload_facial_class():
    face_class_name = request.form['face_class_name']
    relationship_to_user = request.form['relationship_to_user']
    user_id = request.form['user_id']

    # Insert into database with a placeholder image folder path
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO facial_classes (face_class_name, relationship_to_user, user_id, image_folder_path) VALUES (?, ?, ?, ?)',
                   (face_class_name, relationship_to_user, user_id, ''))
    conn.commit()
    face_class_id = cursor.lastrowid  # Get the generated ID

    # Create a directory for this facial class ID and save images
    folder_path = os.path.join(UPLOAD_FOLDER, str(face_class_id))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    images = request.files.getlist('images')
    for img in images:
        filename = secure_filename(img.filename)
        img.save(os.path.join(folder_path, filename))

    # Update the database with the actual image folder path
    cursor.execute('UPDATE facial_classes SET image_folder_path = ? WHERE face_class_id = ?', (folder_path, face_class_id))
    conn.commit()
    conn.close()

    return jsonify({'message': 'Facial class uploaded successfully', 'face_class_id': face_class_id})



# endpoint for object detection history retrieval 
@app.route('/object-detection-history/<int:user_id>', methods=['GET'])
def object_detection_history(user_id):
    conn = get_db_connection()
    objects = conn.execute('SELECT * FROM detected_objects WHERE user_id = ?', (user_id,)).fetchall()
    conn.close()
    return jsonify([dict(row) for row in objects])

if __name__ == '__main__':
   app.run(debug=True)
