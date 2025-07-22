import os

dataset_path = "Dataset\PlantVillage"  

# Get class labels from dataset folder names
class_labels = sorted(os.listdir(dataset_path))

# Assign numeric labels to class names
class_mapping = {i: label for i, label in enumerate(class_labels)}

print("Class Labels:", class_mapping)
