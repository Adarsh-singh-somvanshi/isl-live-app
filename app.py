import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
from PIL import Image

# --- Load TensorFlow Lite Interpreter ---
try:
    import tensorflow.lite as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# --- Streamlit Page Setup ---
st.set_page_config(page_title="ISL Detector", page_icon="🖐️")
st.title("🖐️ ISL Gesture Recognition (Stable Version)")

# --- Load Model + Label Map ---
@st.cache_resource
def load_resources():
    with open("label_map.pkl", "rb") as f:
        label_to_idx = pickle.load(f)
        idx_to_label = {v: k for k, v in label_to_idx.items()}

    interpreter = Interpreter(model_path="isl_gesture_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return idx_to_label, interpreter, input_details, output_details

idx_to_label, interpreter, input_details, output_details = load_resources()

# --- MediaPipe Setup ---
mp_holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

sequence = []   # buffer for 32-frame LSTM input

st.subheader("📸 Show your gesture to the camera")
frame = st.camera_input("Keep your gesture steady for 2–3 seconds")

if frame:
    # Convert input frame to CV2 image
    img = np.array(Image.open(frame))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # MediaPipe Holistic processing
    results = mp_holistic.process(img_rgb)

    # --- Extract Features ---
    vec = []

    # Pose (shoulders 11, 12)
    if results.pose_landmarks:
        for idx in [11, 12]:
            lm = results.pose_landmarks.landmark[idx]
            vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0.0] * 6)

    # Hands (left + right)
    for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand:
            for lm in hand.landmark:
                vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0] * 63)

    # Palm (wrist)
    for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand:
            lm = hand.landmark[0]
            vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0] * 3)

    vec = np.array(vec, dtype=np.float32)

    # Add to sequence buffer
    sequence.append(vec)
    sequence = sequence[-32:]

    # --- Predict when sequence becomes 32 frames ---
    if len(sequence) == 32:
        input_data = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0]
        idx = int(np.argmax(output))

        label = idx_to_label.get(idx, "Unknown")
        conf = float(output[idx]) * 100

        st.success(f"**Prediction: {label} ({conf:.1f}%)**")
    else:
        st.info(f"Collecting frames: {len(sequence)}/32… Hold gesture steady.")
