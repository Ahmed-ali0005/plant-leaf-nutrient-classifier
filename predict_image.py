# Copyright 2025 Chaudhry Ahmed Ali Khan
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Step 1: Load the model
model = load_model('best_model.keras')  # or use 'final_efficientnet_model.keras' if preferred

# Step 2: Class labels (update this according to your dataset)
class_labels = sorted(os.listdir("dataset/train"))  # assumes classes are named folders inside dataset/train

# Step 3: Image preprocessing function
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

# Step 4: Prediction function
def predict(img_path):
    img = preprocess_image(img_path)
    preds = model.predict(img)[0]  # get prediction vector
    results = sorted(zip(class_labels, preds), key=lambda x: x[1], reverse=True)

    print(f"\n📷 Prediction for: {img_path}")
    for label, score in results:
        print(f"{label}: {score:.4f}")

# Step 5: Example usage
if __name__ == "__main__":
    # Replace with the actual path of the image you want to test
    test_image_path = "testcase0.jpg"
    
    if os.path.exists(test_image_path):
        predict(test_image_path)
    else:
        print(f"❌ File not found: {test_image_path}")
