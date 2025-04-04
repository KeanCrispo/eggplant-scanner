import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Debugging: Check if TensorFlow is installed
os.system("pip list | grep tensorflow")

# Manually install TensorFlow if not found (optional)
try:
    import tensorflow as tf
except ModuleNotFoundError:
    os.system("pip install tensorflow")
    import tensorflow as tf

# Load the trained model
MODEL_PATH = "model.h5"

# Function to load the model
@st.cache_resource
def load_trained_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model()

# Streamlit UI
st.title("Eggplant Disease Classification")
st.write("Upload an image to classify the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims for model

    # Make prediction
    if model:
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        # Define class labels (adjust based on your dataset)
        class_labels = ["Healthy", "Multiple Diseases", "Fruit Rot", "Leaf Bright"]
        result = class_labels[predicted_class]

        st.write(f"**Prediction:** {result}")
    else:
        st.error("Model could not be loaded. Check logs.")
