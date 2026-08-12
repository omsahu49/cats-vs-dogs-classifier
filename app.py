import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Cats vs Dogs AI Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

st.title("🐱🐶 Cats vs Dogs AI Classifier")
st.write("Upload an image of a cat or a dog to classify it using AI.")

# Model File Path (Check exact model file name in your repo)
MODEL_PATH = "model.h5"  # Agar .keras format h, to "model.keras" ya sahi name likhna

@st.cache_resource
def load_classifier_model():
    # Searching for available model files in directory
    possible_models = [MODEL_PATH, "model.keras", "cats_vs_dogs.h5", "best_model.h5"]
    found_model = None
    
    for m_path in possible_models:
        if os.path.exists(m_path):
            found_model = m_path
            break
            
    if found_model:
        try:
            return tf.keras.models.load_model(found_model, compile=False), None
        except Exception as e:
            return None, str(e)
    else:
        return None, "No model file found in repository root. Please upload your model file (.h5 or .keras)."

model, load_error = load_classifier_model()

uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Classify Image", type="primary"):
        if model is None:
            st.error(f"Error: {load_error}")
        else:
            with st.spinner("Classifying image..."):
                img_resized = image.resize((150, 150)) # Model input size ke according adjust kar lo
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                img_batch = np.expand_dims(img_array, axis=0)

                raw_pred = float(model.predict(img_batch, verbose=0)[0][0])
                
                st.divider()
                if raw_pred > 0.5:
                    confidence = raw_pred * 100
                    st.success(f"### 🐶 Dog Detected (Confidence: {confidence:.2f}%)")
                else:
                    confidence = (1.0 - raw_pred) * 100
                    st.success(f"### 🐱 Cat Detected (Confidence: {confidence:.2f}%)")
