import os
import random
from PIL import Image, ImageOps, ImageEnhance

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

def augment_images(images, directory_path):
    """Augment a list of images until there are 100 images in total and save them."""
    augmented_images = []
    while len(augmented_images) < 100:
        for image in images:
            if len(augmented_images) >= 100:
                break
            augmented_images.append(augment_image(image))

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for i, augmented_image in enumerate(augmented_images):
        augmented_image.save(os.path.join(directory_path, f'aug_img_{i}.jpg'))
