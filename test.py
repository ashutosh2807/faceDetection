import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model


# # Load the trained model
model = load_model('celebrities.h5')


# Load an image for inference
image_path = 'image.jpeg'
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Preprocess the image
image = image / 255.0  # Normalize pixel values

# Perform inference
predictions = model.predict(image)
predicted_class = np.argmax(predictions[0])
print("Predicted class:", predicted_class)