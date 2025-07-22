import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import json
from PIL import Image

# Load the trained model
model_path = "models/smart_irrigation_cnn.h5"
model = tf.keras.models.load_model(model_path)

# Load class labels
with open("models/class_indices.json", "r") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

# Custom class label mapping (for better readability)
custom_labels = {
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

# Moisture logic
def get_moisture_level(predicted_label):
    if "healthy" in predicted_label:
        return 60
    elif "Early_blight" in predicted_label or "Bacterial_spot" in predicted_label:
        return 25
    else:
        return 85

def decide_irrigation(moisture_level):
    if moisture_level < 30:
        return "💧 Irrigation Needed (Dry Soil)"
    elif 30 <= moisture_level <= 70:
        return "✅ Optimal Moisture - No Irrigation Required"
    else:
        return "🚫 Wet Soil - No Irrigation Required"

# Streamlit UI
st.set_page_config(page_title="Smart Irrigation System", layout="centered")
st.title("🌿 Smart Irrigation System") 

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    
    # Preprocess image
    img = Image.open(uploaded_file)
    img = img.resize((128, 128))
    img = np.array(img)
    
    if img.shape[-1] == 4:
        img = img[:, :, :3]  # Remove alpha channel if present

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    predicted_label = custom_labels.get(predicted_class, "Unknown")

    # Get moisture and decision
    moisture_level = get_moisture_level(predicted_label)
    decision = decide_irrigation(moisture_level)

    # Show results
    st.markdown("### 🧠 Prediction Results")
    st.write(f"**Predicted Class:** {predicted_label}")
    st.write(f"**Soil Moisture Level:** {moisture_level:.2f}%")
    st.success(f"**Irrigation Decision:** {decision}")
