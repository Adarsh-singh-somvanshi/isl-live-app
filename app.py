import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
import pickle
import threading
import asyncio
from collections import deque
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

# --- 1. MONKEY PATCH (Fixes the asyncio crash) ---
# This suppresses the "NoneType object has no attribute call_exception_handler" error
original_del = asyncio.Future.__del__

def new_del(self):
    try:
        original_del(self)
    except (AttributeError, TypeError):
        pass # Ignore the specific crash on teardown

asyncio.Future.__del__ = new_del
# ------------------------------------------------

# --- 2. ROBUST IMPORT SECTION ---
import tensorflow as tf
try:
    Interpreter = tf.lite.Interpreter
except AttributeError:
    try:
        import tensorflow.lite as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

# --- 3. PAGE CONFIG ---
st.set_page_config(page_title="ISL Detector", page_icon="🖐️", layout="wide")
st.title("🖐️ ISL Gesture Recognition")

# --- 4. CONSTANTS & RESOURCES ---
SEQUENCE_LENGTH = 32
FEATURE_SIZE = 138

@st.cache_resource
def load_resources():
    Interpreter = tf.lite.Interpreter # Re-declare for safety
    try:
        with open("label_map.pkl", "rb") as f:
            label_to_idx = pickle.load(f)
            idx_to_label = {v: k for k, v in label_to_idx.items()}
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return None, None

    try:
        interpreter = Interpreter(model_path="isl_gesture_model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        st.error(f"Error loading TFLite model: {e}")
        return None, None
        
    return idx_to_label, (interpreter, input_details, output_details)

resources = load_resources()

# --- 5. VIDEO PROCESSOR ---
class TFLiteProcessor(VideoProcessorBase):
    def __init__(self):
        self.buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.last_pred_text = "Waiting..."
        self.last_confidence = 0.0
        self.lock = threading.Lock()
        
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0, # Lower complexity = faster
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_features(self, results):
        vec = []
        if results.pose_landmarks:
            for idx in [11, 12]:
                lm = results.pose_landmarks.landmark[idx]
                vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0]*3*2)

        for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand:
                for lm in hand.landmark:
                    vec.extend([lm.x, lm.y, lm.z])
            else:
                vec.extend([0.0]*3*21)

        if results.left_hand_landmarks:
            lm = results.left_hand_landmarks.landmark[0]
            vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0]*3)
        if results.right_hand_landmarks:
            lm = results.right_hand_landmarks.landmark[0]
            vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0]*3)

        return np.array(vec, dtype=np.float32) if len(vec) == FEATURE_SIZE else np.zeros(FEATURE_SIZE, dtype=np.float32)

    def recv(self, frame):
        idx_to_label, tflite_data = resources
        if idx_to_label is None or tflite_data is None:
            return frame.to_ndarray(format="bgr24")

        interpreter, input_details, output_details = tflite_data

        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.holistic.process(img_rgb)

        # Draw visuals
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(img, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1))
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        features = self.extract_features(results)
        self.buffer.append(features)

        if len(self.buffer) == SEQUENCE_LENGTH:
            with self.lock:
                input_data = np.expand_dims(np.array(self.buffer, dtype=np.float32), axis=0)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                idx = int(np.argmax(output_data))
                self.last_pred_text = idx_to_label.get(idx, "Unknown")
                self.last_confidence = float(output_data[idx]) * 100

        cv2.rectangle(img, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(img, f"{self.last_pred_text} ({self.last_confidence:.1f}%)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 6. WEBRTC STREAMER (Connection Fix) ---
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="isl-live",
    video_processor_factory=TFLiteProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
        "video": {"width": 480, "height": 480}, # Keeping it stable
        "audio": False
    },
    async_processing=True,
)
