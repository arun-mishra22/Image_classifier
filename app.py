import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import sys

# Reconfigure output for UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Streamlit header
st.header('Image Classification Model')

# Load pre-trained model
model = load_model('./Image_classify.keras')

# Class names
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 
            'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 
            'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 
            'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 
            'sweetpotato', 'tomato', 'turnip', 'watermelon']

# Set image dimensions
image_height = 180
image_width = 180

# Drag-and-drop image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file)
    image = image.resize((image_height, image_width))  # Resize image for the model
    
    # Convert image to array
    img_array = tf.keras.utils.img_to_array(image)
    
    # Expand dimensions for batch processing (model expects batch size as the first dimension)
    img_bat = tf.expand_dims(img_array, 0)

    # Make prediction
    predictions = model.predict(img_bat)
    score = tf.nn.softmax(predictions[0])

    # Display uploaded image
    st.image(image, caption="Uploaded Image", width=200)
    
    # Show prediction and accuracy
    st.write(f'Prediction: **{data_cat[np.argmax(score)]}**')
    st.write(f'Confidence: **{np.max(score) * 100:.2f}%**')
