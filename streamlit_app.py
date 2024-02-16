import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet import preprocess_input
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import boto3
import os

boto3.setup_default_session(
    aws_access_key_id='AKIA2FLNUHZAI6NRYU3J',
    aws_secret_access_key='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',
    region_name='us-east-1'
)

# Initialize a boto3 S3 client
s3 = boto3.client('s3')

# Bucket name
bucket_name = 'chestct'

# Object key (file name in S3)
object_key = 'model.h5'

# Local file name to save the downloaded file
local_file_name = 'model.h5'

# Download the file
try:
    s3.download_file(bucket_name, object_key, local_file_name)
    print("Download successful")
except Exception as e:
    print("Error occurred:", e)
    
# Load the model
model = load_model('model.h5')

def preprocess_image(img):
    # Resize the image to match the model's expected input size
    img = img.resize((224, 224))

    # Convert the image to RGB if it's not already
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Convert the PIL Image to a numpy array
    img_array = img_to_array(img)

    # Apply the same preprocessing as during training
    img_array = preprocess_input(img_array)

    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


st.title("Chest X-Ray Image Classifier")

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Assuming your model outputs a softmax layer, get the class with the highest probability
    class_names = ['AdenoCarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']  # Replace with your actual class names
    class_index = np.argmax(prediction)
    st.write(class_index)
    predicted_class = class_names[class_index]

    # Show the result
    st.write(f"Prediction: {predicted_class} (Class {class_index})")
