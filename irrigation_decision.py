import tensorflow as tf
import numpy as np
import cv2
import os

# Load trained model
model = tf.keras.models.load_model("models/smart_irrigation_cnn.h5")

# Define image path 
image_path = "test_images\sample3.jpg"

# Check if image exists
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

# Define class labels based on dataset categories
class_labels = {
    0: "Pepper__bell___Bacterial_spot",
    1: "Pepper__bell___healthy",
    2: "Potato___Early_blight",
    3: "Potato___Late_blight",
    4: "Potato___healthy",
    5: "Tomato_Bacterial_spot",
    6: "Tomato_Early_blight",
    7: "Tomato_Late_blight",
    8: "Tomato_Leaf_Mold",
    9: "Tomato_Septoria_leaf_spot",
    10: "Tomato_Spider_mites_Two_spotted_spider_mite",
    11: "Tomato__Target_Spot",
    12: "Tomato__Tomato_YellowLeaf__Curl_Virus",
    13: "Tomato__Tomato_mosaic_virus",
    14: "Tomato_healthy"
}

predicted_label = class_labels.get(predicted_class, "Unknown")

# Simulate moisture prediction based on class (adjust logic as needed)
if "healthy" in predicted_label:
    moisture_level = 60  # Assume optimal moisture for healthy plants
elif "Early_blight" in predicted_label or "Bacterial_spot" in predicted_label:
    moisture_level = 25  # Assume dry soil needing irrigation
else:
    moisture_level = 85  # Assume wet soil

# Decide irrigation based on predicted moisture level
def decide_irrigation(moisture_level):
    if moisture_level < 30:
        return " Irrigation Needed (Dry Soil)"
    elif 30 <= moisture_level <= 70:
        return " Optimal Moisture - No Irrigation Required"
    else:
        return " Wet Soil - No Irrigation Required"

decision = decide_irrigation(moisture_level)

# Print results
print(f" Predicted Class: {predicted_label}")
print(f" Soil Moisture Level: {moisture_level:.2f}%")
print(f" Irrigation Decision: {decision}")
