import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
from PIL import Image
import base64

try:
    import tensorflow.lite as tflite
    Interpreter = tflite.Interpreter
except:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

st.set_page_config(page_title="Real-Time ISL", page_icon="🖐️")
st.title("🖐️ Real-Time ISL Gesture Recognition")

# Load model + labels
@st.cache_resource
def load_resources():
    with open("label_map.pkl", "rb") as f:
        label_to_idx = pickle.load(f)
        idx_to_label = {v: k for k, v in label_to_idx.items()}

    interpreter = Interpreter("isl_gesture_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return idx_to_label, interpreter, input_details, output_details

idx_to_label, interpreter, input_details, output_details = load_resources()

# Mediapipe
mp_holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

sequence = []
output_placeholder = st.empty()

# JS live webcam feed
st.markdown("""
    <style>
        video {transform: scaleX(-1);}
    </style>
    <video id="video" width="640" height="480" autoplay></video>
    <script>
    const video = document.getElementById('video');

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => { video.srcObject = stream; });

    function captureFrame() {
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, 640, 480);
        return canvas.toDataURL('image/jpeg');
    }

    async function sendFrame() {
        const imgData = captureFrame();
        await fetch('/live_frame', {
            method: 'POST',
            body: imgData
        });
        requestAnimationFrame(sendFrame);
    }
    requestAnimationFrame(sendFrame);
    </script>
""", unsafe_allow_html=True)
