import tensorflow as tf
import numpy as np
import cv2
import os
import json

# Load trained model
model_path = "models\smart_irrigation_cnn.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = tf.keras.models.load_model(model_path)

# Load class labels
class_indices_path = "models/class_indices.json"
if not os.path.exists(class_indices_path):
    raise FileNotFoundError(f"Class indices file not found at {class_indices_path}")

with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

# Reverse the dictionary to get class names from indices
class_labels = {v: k for k, v in class_indices.items()}

# Define image path
image_path = "test_images\sample1.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Test image not found at {image_path}")

# Load and preprocess image
img = cv2.imread(image_path)
img = cv2.resize(img, (128, 128))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict class
prediction = model.predict(img)
predicted_class = np.argmax(prediction)

# Get class label
predicted_label = class_labels.get(predicted_class, "Unknown")

print(f"Predicted Class: {predicted_label}")
