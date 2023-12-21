# Import necessary libraries
import pandas as pd
import numpy as np
import os
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# Tensorflow and Keras Modules
from keras.utils import load_img
from keras.models import load_model

# Set the directory
BASE_DIR = 'UTKFace/'

# Initialize lists to store image paths, age labels, and gender labels
image_paths = []
age_labels = []
gender_labels = []

# Loop through the dataset to collect image paths, age labels, and gender labels
for filename in os.listdir(BASE_DIR):
    image_path = os.path.join(BASE_DIR, filename)
    temp = filename.split('_')
    age = int(temp[0])
    gender = int(temp[1])
    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)

# Create a DataFrame
df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_paths, age_labels, gender_labels

# Change the label of the gender
gender_dict = {0: 'Male', 1: 'Female'}
df['gender'] = df['gender'].map(gender_dict)

# Function to extract features
def extract_features(images):
    features = []
    for image in images:
        img = load_img(image, grayscale=True)
        img = img.resize((128, 128), Image.ANTIALIAS)
        img = np.array(img)
        features.append(img)

    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)
    return features

# Extract features and normalize
X = extract_features(df['image']) / 255.0

# Load the trained model
model = load_model("gender_age_detection_model.h5")

# Make predictions
predictions = model.predict(X)

# Extract gender and age predictions
gender_predictions = np.round(predictions[0]).astype(int)
age_predictions = np.round(predictions[1]).astype(int)

# Map gender predictions to labels
predicted_genders = np.vectorize(gender_dict.get)(gender_predictions.flatten())

# Display predictions
for i in range(len(df)):
    print(f"Actual - Gender: {df['gender'][i]}, Age: {df['age'][i]} | Predicted - Gender: {predicted_genders[i]}, Age: {age_predictions[i]}")
