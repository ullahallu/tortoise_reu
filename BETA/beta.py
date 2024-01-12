# SCRIPT FOR UPDATING THE DATABASE MANUALLY
import sqlite3
from datetime import datetime

# Database file
DATABASE = 'tortoise_0002.db'

# Function to get database connection
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    return conn

# Function to add a new user
def add_new_user(name, credentials):
    conn = get_db_connection()
    conn.execute('INSERT INTO system_users (user_name, credentials) VALUES (?, ?)', (name, credentials))
    conn.commit()
    conn.close()
    print(f"User '{name}' added successfully.")

# Function to add a new object detection
def add_new_object_detection(object_class, gps_location, user_id, image_path):
    timestamp = datetime.now()
    conn = get_db_connection()
    conn.execute('INSERT INTO detected_objects (object_class, gps_location, user_id, image_path, timestamp) VALUES (?, ?, ?, ?, ?)', 
                 (object_class, gps_location, user_id, image_path, timestamp))
    conn.commit()
    conn.close()
    print(f"Object detection '{object_class}' added successfully.")

# Menu for user interaction
def menu():
    while True:
        print("\nMenu:")
        
        print("1. Add New User")
        print("2. Add New Object Detection")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            name = input("Enter user name: ")
            credentials = input("Enter user credentials: ")
            add_new_user(name, credentials)
        elif choice == '2':
            object_class = input("Enter object class: ")
            gps_location = input("Enter GPS location: ")
            user_id = int(input("Enter user ID capturing object: "))
            image_path = input("Enter image path: ")  # Dummy or actual path
            add_new_object_detection(object_class, gps_location, user_id, image_path)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    menu()
