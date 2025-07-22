import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import json

# Define dataset path
dataset_path = "Dataset/PlantVillage"

# Ensure dataset directory exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

# Data augmentation & preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

# Load training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Load validation data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_data.class_indices), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_data, validation_data=val_data, epochs=10)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save model
model.save("models/smart_irrigation_cnn.h5")

# Save class indices for testing
class_indices_path = "models/class_indices.json"
with open(class_indices_path, "w") as f:
    json.dump(train_data.class_indices, f)

print("Training complete. Model and class indices saved successfully!")
