import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import CosineSimilarity
import os

MODEL_PATH = r'Color_model\color-embeded-acc.h5'

if "model" not in st.session_state:
    st.session_state["model"] = load_model(MODEL_PATH)
model =  st.session_state["model"]

# Configure MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, refine_landmarks=True)

# Landmark filters
filtered_hand = list(range(21))
filtered_pose = [11, 12, 13, 14, 15, 16]
filtered_face = [4, 6, 8, 9, 33, 37, 40, 46, 52, 55, 61, 70, 80, 82, 84, 87, 88, 91, 105, 107, 133, 145, 154, 157, 159, 161, 163, 263, 267, 270, 276, 282, 285, 291, 300, 310, 312, 314, 317, 318, 321, 334, 336, 362, 374, 381, 384, 386, 388, 390, 468, 473]
HAND_NUM, POSE_NUM, FACE_NUM = len(filtered_hand), len(filtered_pose), len(filtered_face)

# Threshold
THRESHOLD = 0.8

# Define functions
def get_frame_landmarks(frame):
    """Extract landmarks from a frame."""
    all_landmarks = np.zeros((HAND_NUM * 2 + POSE_NUM + FACE_NUM, 3))

    def get_hands(frame):
        results_hands = hands.process(frame)
        if results_hands.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                if results_hands.multi_handedness[i].classification[0].index == 0:  # Right hand
                    all_landmarks[:HAND_NUM, :] = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
                else:
                    all_landmarks[HAND_NUM:HAND_NUM * 2, :] = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

    def get_pose(frame):
        results_pose = pose.process(frame)
        if results_pose.pose_landmarks:
            all_landmarks[HAND_NUM * 2:HAND_NUM * 2 + POSE_NUM, :] = np.array(
                [(lm.x, lm.y, lm.z) for lm in results_pose.pose_landmarks.landmark])[filtered_pose]

    def get_face(frame):
        results_face = face_mesh.process(frame)
        if results_face.multi_face_landmarks:
            all_landmarks[HAND_NUM * 2 + POSE_NUM:, :] = np.array(
                [(lm.x, lm.y, lm.z) for lm in results_face.multi_face_landmarks[0].landmark])[filtered_face]

    get_hands(frame)
    get_pose(frame)
    get_face(frame)

    return all_landmarks

def extract_embedding(video_landmarks):
    """Create an embedding from landmarks."""
    video_landmarks = np.array(video_landmarks)
    target_shape = (60, 100, 3)

    if video_landmarks.shape[0] < target_shape[0]:
        padding = target_shape[0] - video_landmarks.shape[0]
        video_landmarks = np.pad(video_landmarks, ((0, padding), (0, 0), (0, 0)), mode='constant', constant_values=0)
    else:
        video_landmarks = video_landmarks[:target_shape[0], :, :]

    video_landmarks = np.reshape(video_landmarks, (1, *target_shape))
    embedding = model.predict(video_landmarks)
    return embedding

def calculate_cosine_similarity(embedding1, embedding2):
    """Calculate Cosine Similarity between two embeddings."""
    cosine_similarity = CosineSimilarity()
    return cosine_similarity(embedding1, embedding2).numpy()

# Save reference embedding if not exists
REFERENCE_EMBEDDING_PATH = "reference_embedding.npy"
VIDEO_PATH = r"Color_video\164951232112037-CONGRATULATIONS.mp4"

if not os.path.exists(REFERENCE_EMBEDDING_PATH):
    cap = cv2.VideoCapture(VIDEO_PATH)
    ref_landmarks = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ref_landmarks.append(get_frame_landmarks(frame_rgb))
    cap.release()

    reference_embedding = extract_embedding(ref_landmarks)
    np.save(REFERENCE_EMBEDDING_PATH, reference_embedding)
else:
    reference_embedding = np.load(REFERENCE_EMBEDDING_PATH)


st.set_page_config(page_title="Sign Language Learning", layout="wide")

if "match_detected" not in st.session_state:
    st.session_state["match_detected"] = False

# Sidebar with Roadmap
st.sidebar.title("Learning Roadmap")
roadmap_videos = {
    "Chapter 1": [VIDEO_PATH, VIDEO_PATH, VIDEO_PATH],
    "Chapter 2": [VIDEO_PATH, VIDEO_PATH, VIDEO_PATH]
}
selected_chapter = st.sidebar.selectbox("Select a Chapter", list(roadmap_videos.keys()))
selected_lesson = st.sidebar.selectbox("Select a Lesson", range(1, len(roadmap_videos[selected_chapter]) + 1))
lesson_video_path = roadmap_videos[selected_chapter][selected_lesson - 1]


col1, col2 = st.columns(2)

with col1:
    st.title("Example")
    st.video(lesson_video_path)

with col2:
    st.title("Practice")

    class SignLanguageTransformer(VideoTransformerBase):
        def __init__(self):
            self.reference_embedding = reference_embedding
            self.user_landmarks = []

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            landmarks = get_frame_landmarks(img_rgb)
            self.user_landmarks.append(landmarks)

            if len(self.user_landmarks) > 60:  # Process 60 frames
                user_embedding = extract_embedding(self.user_landmarks)
                similarity = calculate_cosine_similarity(self.reference_embedding, user_embedding)

                if similarity > THRESHOLD:
                    cv2.putText(img, "Match!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    st.session_state["match_detected"] = True
                else:
                    cv2.putText(img, "Keep Practicing", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    st.session_state["match_detected"] = False

                self.user_landmarks = []  

            return img

    webrtc_streamer(key="sign_language_practice", video_transformer_factory=SignLanguageTransformer)


if st.session_state["match_detected"]:
    st.success("Chúc mừng! Bạn đã thực hiện đúng cử chỉ!")
