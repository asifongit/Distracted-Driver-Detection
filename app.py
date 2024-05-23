import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf

# Load the trained model
model = load_model(r'C:\Users\DELL\Documents\ANN LAB\Assignment\final project\model_v1.h5')

# Define activity map
activity_map = {
    'c0': 'Safe driving', 
    'c1': 'Texting - right', 
    'c2': 'Talking on the phone - right', 
    'c3': 'Texting - left', 
    'c4': 'Talking on the phone - left', 
    'c5': 'Operating the radio', 
    'c6': 'Drinking', 
    'c7': 'Reaching behind', 
    'c8': 'Hair and makeup', 
    'c9': 'Talking to passenger'
}

def get_cv2_image(img_bytes, img_rows, img_cols, color_type=3):
    img = cv2.imdecode(np.fromstring(img_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
    if color_type == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_rows, img_cols))
    return img

def main():
    st.title("Driver Behavior Classification")
    st.write("Upload an image and the model will predict the driver's behavior.")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        img = get_cv2_image(uploaded_file, 64, 64, 1)
        st.image(img, channels="GRAY", caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")
        
        # Prepare image for prediction
        img = img.reshape(1, 64, 64, 1)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_activity = activity_map[f'c{predicted_class}']

        st.write(f"Predicted Activity: {predicted_activity}")
    
    st.write("Note: Ensure the uploaded image is a clear view of the driver for accurate predictions.")

if __name__ == '__main__':
    main()
