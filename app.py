import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
import pickle
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

# --- 1. IMPORT TENSORFLOW SAFELY ---
try:
    import tensorflow.lite as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# --- 2. PAGE SETUP ---
st.set_page_config(page_title="ISL Detector", page_icon="🖐️")
st.title("🖐️ ISL Gesture Recognition")

# --- 3. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        # Load Labels
        with open("label_map.pkl", "rb") as f:
            label_to_idx = pickle.load(f)
            idx_to_label = {v: k for k, v in label_to_idx.items()}
        
        # Load Model
        interpreter = Interpreter(model_path="isl_gesture_model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return idx_to_label, (interpreter, input_details, output_details)
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

resources = load_resources()

# --- 4. PROCESSOR CLASS ---
class TFLiteProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1, # Must be 1 to avoid download permission error
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.sequence = []
        self.last_prediction = "Waiting..."
        self.last_conf = 0.0

    def extract_features(self, results):
        vec = []
        # Pose
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
        for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand:
                lm = hand.landmark[0]
                vec.extend([lm.x, lm.y, lm.z])
            else:
                vec.extend([0.0]*3)
        
        return np.array(vec, dtype=np.float32)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if resources is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        idx_to_label, (interpreter, input_details, output_details) = resources

        # 1. MediaPipe Processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)

        # 2. Draw Landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(img, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(img, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        # 3. Prediction Logic
        features = self.extract_features(results)
        self.sequence.append(features)
        self.sequence = self.sequence[-32:] # Keep last 32 frames

        if len(self.sequence) == 32:
            try:
                input_data = np.expand_dims(np.array(self.sequence, dtype=np.float32), axis=0)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                
                idx = int(np.argmax(output_data))
                self.last_prediction = idx_to_label.get(idx, "Unknown")
                self.last_conf = float(output_data[idx]) * 100
            except Exception:
                pass

        # 4. Draw Text
        cv2.rectangle(img, (0,0), (640, 40), (0,0,0), -1)
        cv2.putText(img, f"{self.last_prediction} ({self.last_conf:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. WEBRTC SETUP ---
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="isl",
    video_processor_factory=TFLiteProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False}
)
