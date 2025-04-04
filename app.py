import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import cv2
import base64

# Function to set background
def set_background(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                height: 100vh;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error setting background image: {e}")

# Call background function
set_background("background.jpg")  # Make sure the image is in the correct path

# Load the model and cache
@st.cache_resource
def load_model(path):
    # Xception Model
    xception_model = tf.keras.models.Sequential([
        tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # DenseNet Model
    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Ensemble
    inputs = tf.keras.Input(shape=(512, 512, 3))
    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)
    outputs = tf.keras.layers.average([densenet_output, xception_output])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.load_weights(path)
    return model

# Load model
model = load_model("eggplant_model.h5")

# Hide Streamlit branding
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Styling text to pop from the background
st.markdown("""
    <style>
    h1, h2, h3, p {
        color: white;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    }
    .stTitle {
        font-size: 36px;
        text-align: center;
        color: white;  /* Added to set title text color to white */
    }
    </style>
""", unsafe_allow_html=True)

# Styling the button to make font color black
st.markdown("""
    <style>
    .stButton > button {
        color: black !important;
        font-weight: bold;
        background-color: #e0e0e0;
        border: 2px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

# Set title color to white
st.markdown("""
    <style>
    h1 {
        color: white !important;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Title with white color
st.markdown("<h1>Eggplant Disease Detection</h1>", unsafe_allow_html=True)

st.write("Upload an image or scan live using your camera to check eggplant health.")

option = st.radio("Select Input Method:", ('Upload Image', 'Live Camera Scan'))

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image, caption="Captured Frame", use_container_width=True)
        image = clean_image(image)
        predictions, predictions_arr = get_prediction(model, image)
        result = make_results(predictions, predictions_arr)
        st.write(f"The eggplant is **{result['status']}** with **{result['prediction']}** prediction.")

elif option == 'Live Camera Scan':
    st.write("Enable webcam below and click 'Analyze Last Frame'")

    # Store frame in transformer
    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.latest_frame = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)  # Mirror the camera input
            self.latest_frame = img  # Save latest frame
            return img

    ctx = webrtc_streamer(
        key="mirror-cam",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}}
        },
        async_processing=True,
    )

    if ctx.video_transformer:
        if st.button("Analyze Last Frame"):
            frame = ctx.video_transformer.latest_frame
            if frame is not None:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_rgb)
                st.image(image, caption="Captured Frame", use_container_width=True)
                image = clean_image(image)
                predictions, predictions_arr = get_prediction(model, image)
                result = make_results(predictions, predictions_arr)
                st.write(f"The eggplant {result['status']} with {result['prediction']} prediction.")
            else:
                st.warning("Please wait for the camera to initialize and capture a frame.")
