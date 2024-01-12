# TKinter application 

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Listbox
import requests

# Initialize Tkinter window
root = tk.Tk()
root.title('Tortoise System Interface')

# Function to select images for facial class
def select_images():
    file_paths = filedialog.askopenfilenames(title='Select Images for Facial Class')
    return file_paths

# Function to upload images and relationship for new facial class
def upload_facial_class(images, face_class_name, relationship_to_user, user_id):
    files = [('images', (open(image, 'rb'))) for image in images]
    data = {'face_class_name': face_class_name, 'relationship_to_user': relationship_to_user, 'user_id': user_id}
    
    response = requests.post('http://127.0.0.1:5000/upload-facial-class', files=files, data=data)
    return response.json()

# UI components for image selection and relationship entry for facial class
def facial_class_upload_page():
    upload_window = tk.Toplevel(root)
    upload_window.title("Upload Facial Class")

    face_class_name = simpledialog.askstring("Input", "Enter face class name")
    relationship_to_user = simpledialog.askstring("Input", "Enter relationship to user")
    user_id = simpledialog.askinteger("Input", "Enter user ID")

    selected_images = select_images()  # Get selected images

    if selected_images:
        response = upload_facial_class(selected_images, face_class_name, relationship_to_user, user_id)
        messagebox.showinfo("Response", response)

# Function to fetch object detection history
def fetch_object_detection_history(user_id):
    response = requests.get(f'http://127.0.0.1:5000/object-detection-history/{user_id}')
    if response.ok:
        return response.json()
    else:
        messagebox.showerror("Error", "Failed to fetch data")
        return []

# Function to update the listbox with object detection history
def update_history_listbox(listbox, user_id):
    listbox.delete(0, tk.END)  # Clear existing entries
    history = fetch_object_detection_history(user_id)
    for item in history:
        listbox.insert(tk.END, f"{item['object_class']} at {item['gps_location']} on {item['timestamp']}")

# UI for object detection history
def object_detection_history_page():
    history_window = tk.Toplevel(root)
    history_window.title("Object Detection History")

    user_id_entry = tk.Entry(history_window)
    user_id_entry.pack()

    history_listbox = Listbox(history_window)
    history_listbox.pack()

    update_button = tk.Button(history_window, text="Fetch History", command=lambda: update_history_listbox(history_listbox, user_id_entry.get()))
    update_button.pack()

# Buttons on main window to access different functionalities
facial_class_button = tk.Button(root, text="Upload Facial Class", command=facial_class_upload_page)
facial_class_button.pack()

history_button = tk.Button(root, text="Object Detection History", command=object_detection_history_page)
history_button.pack()

# Start the Tkinter event loop
root.mainloop()
