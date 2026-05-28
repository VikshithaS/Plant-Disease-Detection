import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

# Function to preprocess image
def preprocess_image(img_path):

    # Load image
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert image to array
    img_array = image.img_to_array(img)

    # Normalize image
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

print("Preprocessor ready!")