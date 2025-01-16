import streamlit as st 
import tensorflow as tf
import numpy as np
import cv2
import os

# Define the model path
MODEL_PATH = "C:\Users\HP\OneDrive\Desktop\PlantDiseaseDetection (1)"


def load_model(): 
    return
tf.keras.models.load_model(MODEL_PATH)

    img = cv2.imread(image_path)
    H, W, C = 224, 224, 3
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    img = img.astype("float32") / 255.0
    img = img.reshape(1, H, W, C)

    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

# Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION", "ABOUT DEVELOPER"])

# import Image from pillow to open images
from PIL import Image
img = Image.open("C:/Users/Lenovo/Downloads/Diseases.png")

# Display image using streamlit
st.image(img)

# Home Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")

    test_image = st.file_uploader("Choose an Image:")

    if test_image is not None:
        # Define the save path
        save_path = os.path.join(os.getcwd(), test_image.name)
        print(save_path)

        # Save the file to the working directory
        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())

    if st.button("Show Image"):
        st.image(test_image, use_column_width=True)

    # Predict button
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_predict(save_path)
        print(result_index)
        class_names = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)_Powdery_mildew',
            'Cherry_(including_sour)_healthy',
            'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)Common_rust',
            'Corn_(maize)_Northern_Leaf_Blight',
            'Corn_(maize)_healthy',
            'Grape___Black_rot',
            'Grape__Esca(Black_Measles)',
            'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange_Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper, bell___Bacterial_spot',
            'Pepper, bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        st.success("Model is Predicting it's a {}".format(class_names[result_index]))

# About Developer Page
elif app_mode == "ABOUT DEVELOPER":
    st.header("About the Developer")
    st.markdown("""
        Developer Name: [Siddharth Kumar]  
        Role: Software Developer and AI Enthusiast  
        Description: Passionate about creating innovative AI solutions to tackle real-world problems. This Plant Disease Detection system is designed to assist farmers in sustainable agriculture practices.
        Contact: [ siddharthinfo001@gmail.com]
    """)