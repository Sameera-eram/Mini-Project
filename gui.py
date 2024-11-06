import streamlit as st
import numpy as np
import cv2
import joblib
import json
import os
import pandas as pd
from PIL import Image
import pywt

# Load model and class dictionaries
model = joblib.load("saved_model.pkl")
with open("class_dictionary.json", "r") as f:
    class_dict = json.load(f)
with open("player_info.json", "r") as f:
    player_info = json.load(f)

reverse_class_dict = {v: k for k, v in class_dict.items()}

# Set background color to black and text color to white
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
        font-family: Monospace
    }
    h1, h2, h3, h4 {
        color: #FFFFFF;
        font-weight: bold;
        font-family: Monospace
    }
    .sport-description {
        font-size: 25px;
        color: #FFFFFF;
        font-family: Monospace
    }
    .achievements {
        font-size: 25px;
        color: #FFFFFF;
        font-family: Monospace
    }
    .stFileUpload > label {
        color: white;
        font-family: sMonospace
    }
    /* Table styling */
    table {
        color: #FFFFFF;
        border: 5px solid white;
        font-family: Monospace
    }
    th {
        color: white;
        font-weight: bold;
        border-bottom: 5px solid white;
        font-family: Monospace
    }
    td {
        color: white;
        border: 1px solid white;
        font-family: Monospace
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function for preprocessing image
def preprocess_image(uploaded_image):
    img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        st.error("No face detected. Please upload an image with a clear face.")
        return None, None
    
    (x, y, w, h) = faces[0]
    roi_color = img[y:y+h, x:x+w]
    
    return img, roi_color

# Wavelet transformation
def w2d(img, mode='haar', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray) / 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H = np.uint8(imArray_H * 255)
    return imArray_H

# Model preprocessing
def preprocess_for_model(roi_color):
    scalled_raw_img = cv2.resize(roi_color, (32, 32))
    img_har = w2d(roi_color, 'db1', 5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((scalled_raw_img.reshape(32*32*3, 1), scalled_img_har.reshape(32*32, 1)))
    return combined_img.reshape(1, -1).astype(float)

# Load player image
def load_player_image(player_name):
    image_path = f"./images/{player_name}.jpg"
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        st.warning("No image found for this player.")
        return None

# UI setup
st.title("🏆 Sports Person Image Classification")
st.markdown("<h4 style='color: #FFFFFF;'>Identify the sports person and learn about them!</h4>", unsafe_allow_html=True)

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=False, width=300)  
    img, roi_color = preprocess_image(uploaded_image)

    if roi_color is not None:
        if st.button("Classify"):
            processed_image = preprocess_for_model(roi_color)
            probabilities = model.predict_proba(processed_image)[0]
            predicted_index = np.argmax(probabilities)
            predicted_name = reverse_class_dict[predicted_index]

            # Display predicted player and their info
            st.subheader(f"Predicted Player: {predicted_name}")
            player_image = load_player_image(predicted_name)
            if player_image:
                st.image(player_image, caption=predicted_name, use_column_width=False, width=300)
            
            if predicted_name in player_info:
                info = player_info[predicted_name]
                st.markdown(f"<div class='sport-description'>⭐&nbsp;Sport:&nbsp; {info['sport']} </div>", unsafe_allow_html=True)
                st.markdown(f"<div class='sport-description'>⭐&nbsp;{info['description']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='achievements'> ⭐&nbsp;{info['achievements']}</div>", unsafe_allow_html=True)

            # Display top 5 probabilities
            st.subheader("Top 5 Prediction Probabilities")
            top_indices = np.argsort(probabilities)[-5:][::-1]  
            top_players = [reverse_class_dict[i] for i in top_indices]
            top_probs = probabilities[top_indices]

            prob_df = pd.DataFrame({
                "Player": top_players,
                "Probability": top_probs
            })
            st.table(prob_df)
