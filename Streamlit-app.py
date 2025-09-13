import cv2
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

@st.cache_resource
# load the model
def load_classification_model():
    return load_model('cat_dog_cnn_model.keras')

try:
    model = load_classification_model()
    # st.success('Model Loaded Successfully')
except Exception as e:
    st.error('Error loading model: ',e)
    
# title of the web page
st.title('Dog Cat Classification')

# Upload file
uploaded_file = st.file_uploader('Uplod cat or dog image:', type = ['jpg','jpeg','png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption = 'Uploaded Image', width = 500)
    st.success('Image Uploaded Successfully')
    
    img = img.resize((256,256))
    img_array = image.img_to_array(img)/255.
    
    img = np.expand_dims(img_array, axis = 0) #(1,256,256,3)
    
    if st.button('Predict Image'):
        prediction = model.predict(img)
        if prediction < 0.5:
            st.success(f"The Uploaded image 'Cat', Confidence = {prediction}")
        else:
            st.success(f"The Uploaded Image 'Dog', Confidence = {prediction}")
        
        
# Instruction
st.sidebar.header('Instructions')
st.sidebar.info(
    "1. Upload an image of a dog or cat.\n"
    '2. Click the predict button.\n'
    '3. View the prediction result.'
)

# About
st.sidebar.header('About')
st.sidebar.info(
    "This app uses a CNNs model trained on a dogs and Cats dataset.\n"
    "The model was trained using Tensorflow/Keras."
)