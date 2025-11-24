import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
import pickle
import threading
from collections import deque
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

# --- 1. RELIABLE IMPORT SECTION ---
# We use the standard import which is most stable in TF 2.15
import tensorflow as tf
try:
    Interpreter = tf.lite.Interpreter
except AttributeError:
    # Robust fallback for some Windows environments
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        st.error("TensorFlow Lite is missing. Please run: pip install tensorflow")

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="ISL Detector", page_icon="🖐️", layout="wide")
st.title("🖐️ ISL Gesture Recognition (HD Mode)")
st.write("Real-time detection using TFLite. Ensure good lighting for best results.")

# --- 3. CONSTANTS & RESOURCES ---
SEQUENCE_LENGTH = 32
FEATURE_SIZE = 138

@st.cache_resource
def load_resources():
    # Load Labels
    try:
        with open("label_map.pkl", "rb") as f:
            label_to_idx = pickle.load(f)
            idx_to_label = {v: k for k, v in label_to_idx.items()}
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return None, None

    # Load TFLite Model
    try:
        # Using the Interpreter class we safely imported above
        interpreter = Interpreter(model_path="isl_gesture_model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        st.error(f"Error loading TFLite model: {e}")
        return None, None
        
    return idx_to_label, (interpreter, input_details, output_details)

resources = load_resources()

# --- 4. VIDEO PROCESSOR ---
class TFLiteProcessor(VideoProcessorBase):
    def __init__(self):
        self.buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.last_pred_text = "Initializing..."
        self.last_confidence = 0.0
        self.lock = threading.Lock() # Prevents threading conflicts
        
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=False
        )

    def extract_features(self, results):
        vec = []
        # Pose (Shoulders)
        if results.pose_landmarks:
            for idx in [11, 12]:
                lm = results.pose_landmarks.landmark[idx]
                vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0]*3*2)

        # Hands
        for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand:
                for lm in hand.landmark:
                    vec.extend([lm.x, lm.y, lm.z])
            else:
                vec.extend([0.0]*3*21)

        # Palms
        if results.left_hand_landmarks:
            vec.extend([results.left_hand_landmarks.landmark[0].x, results.left_hand_landmarks.landmark[0].y, results.left_hand_landmarks.landmark[0].z])
        else:
            vec.extend([0.0]*3)
        if results.right_hand_landmarks:
            vec.extend([results.right_hand_landmarks.landmark[0].x, results.right_hand_landmarks.landmark[0].y, results.right_hand_landmarks.landmark[0].z])
        else:
            vec.extend([0.0]*3)

        return np.array(vec, dtype=np.float32) if len(vec) == FEATURE_SIZE else np.zeros(FEATURE_SIZE, dtype=np.float32)

    def recv(self, frame):
        idx_to_label, tflite_data = resources
        if idx_to_label is None or tflite_data is None:
            return frame

        interpreter, input_details, output_details = tflite_data

        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.holistic.process(img_rgb)

        # Draw Face Mesh (Hiding Face)
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                img, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
            )
        
        # Draw Hands
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img, hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
                )

        # Prediction Logic
        features = self.extract_features(results)
        self.buffer.append(features)

        if len(self.buffer) == SEQUENCE_LENGTH:
            with self.lock: # Thread safety for inference
                input_data = np.expand_dims(np.array(self.buffer, dtype=np.float32), axis=0)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                
                idx = int(np.argmax(output_data))
                conf = float(output_data[idx])
                
                self.last_pred_text = idx_to_label.get(idx, "Unknown")
                self.last_confidence = conf * 100

        # UI Overlay
        cv2.rectangle(img, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(img, f"{self.last_pred_text} ({self.last_confidence:.1f}%)", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. WEBRTC STREAMER (With HD Settings) ---
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Force HD Resolution (1280x720)
webrtc_streamer(
    key="isl-hd",
    video_processor_factory=TFLiteProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
        "video": {
            "width": {"min": 1280, "ideal": 1280, "max": 1920},
            "height": {"min": 720, "ideal": 720, "max": 1080},
        },
        "audio": False
    }
)