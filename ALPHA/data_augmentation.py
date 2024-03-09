# DATA AUGMENTATION SCRIPT

import os
import random
from PIL import Image, ImageOps, ImageEnhance

label_name = "timmy"
def augment_image(image):
    """Apply random augmentation to an image."""
    choice = random.randint(1, 4)
    if choice == 1:
        return image.rotate(random.randint(-30, 30))
    elif choice == 2:
        return ImageOps.mirror(image)
    elif choice == 3:
        return ImageOps.flip(image)
    elif choice == 4:
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(random.uniform(0.5, 1.5))

def augment_images(directory_path):
    """Augment images from a directory until there are 100 images in total."""
    image_files = [f for f in os.listdir(directory_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    augmented_images = []


    while len(augmented_images) < 100:
        for file in image_files:
            if len(augmented_images) >= 100:
                break
            with Image.open(os.path.join(directory_path, file)) as img:
                augmented_images.append(augment_image(img))
  
    new_augmented_directory = f"data/{label_name}_augmented"
    # Create a new directory to save augmented images
    os.makedirs(f"{new_augmented_directory}", exist_ok=True)

    # Save the augmented images
    for i, image in enumerate(augmented_images):
        image.save(f'{new_augmented_directory}/aug_img_{i}.jpg')

# Example usage
directory_path = f"data/{label_name}"  # Replace with your directory path
augment_images(directory_path)
