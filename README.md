#Smart Irrigation System

A ) Introduction

This project implements a Smart Irrigation System using Deep Learning and Computer Vision. It analyzes soil moisture levels and plant health to make irrigation decisions. The model is trained on the PlantVillage dataset and predicts soil conditions using a Convolutional Neural Network (CNN).

B) Project Structure:

1. Dataset: PlantVillage dataset containing images of different plant conditions.

2. train_model.py: Trains a CNN on the dataset and saves the model.

3. test_model.py: Loads the trained model and predicts plant health from images.

4. irrigation_decision.py: Determines whether irrigation is required based on soil moisture levels.

5. models/: Directory where trained models are stored.

C) Steps to Create the Model:

1. Data Preparation

Downloaded the PlantVillage dataset and structured it into training and validation sets.
Used ImageDataGenerator for data augmentation and preprocessing.

2. Model Training (train_model.py)

Built a CNN model with Conv2D, MaxPooling2D, Flatten, and Dense layers.
Compiled using Adam optimizer and categorical cross-entropy loss.
Trained for 10 epochs on the dataset.
Saved the trained model to models/smart_irrigation_cnn.h5.

3. Model Testing (test_model.py)

Loaded the trained CNN model.
Preprocessed input images and made predictions.
Mapped predictions to class labels.

4. Irrigation Decision System (irrigation_decision.py)

Measured soil moisture levels from images.
Applied threshold-based decision-making:

If moisture < 30%: Irrigation Needed.

If moisture 30-70%: Optimal Moisture - No Irrigation.

If moisture > 70%: Wet Soil - No Irrigation.

D) How to Run the Project:

1. Install Dependencies

pip install tensorflow keras numpy opencv-python matplotlib

2. Train the Model

python train_model.py

3. Test the Model

python test_model.py --image path/to/image.jpg

4. Make an Irrigation Decision

python irrigation_decision.py --image path/to/soil_image.jpg

E) Conclusion:

This project automates irrigation decisions based on soil conditions, helping optimize water usage for sustainable agriculture.
