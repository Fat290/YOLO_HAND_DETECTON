import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import CosineSimilarity
import streamlit as st

# Đường dẫn mô hình và video
MODEL_PATH = r'Color_model\color-embeded-acc.h5'
VIDEO_PATH = r"Color_videos\18880274572777278-BRIGHT.mp4"  

# Tải mô hình
model = load_model(MODEL_PATH)

# Cấu hình MediaPipe
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

def get_frame_landmarks(frame):
    """Trích xuất landmarks từ một khung hình."""
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

def extract_embedding(video_landmarks, model, target_shape=(60, 100, 3)):
    """Tạo chuỗi embedding từ landmarks."""
    video_landmarks = np.array(video_landmarks)

    if video_landmarks.shape[0] < target_shape[0]:
        padding = target_shape[0] - video_landmarks.shape[0]
        video_landmarks = np.pad(video_landmarks, ((0, padding), (0, 0), (0, 0)), mode='constant', constant_values=0)
    else:
        video_landmarks = video_landmarks[:target_shape[0], :, :]

    video_landmarks = np.reshape(video_landmarks, (1, *target_shape))
    embedding = model.predict(video_landmarks)
    return embedding

def calculate_cosine_similarity(embedding1, embedding2):
    """Tính toán độ tương đồng Cosine giữa hai vector embedding."""
    cosine_similarity = CosineSimilarity()
    return cosine_similarity(embedding1, embedding2).numpy()

def get_video_embedding(video_path, model):
    """Trích xuất landmarks từ video và tạo embedding."""
    cap = cv2.VideoCapture(video_path)
    all_landmarks = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = get_frame_landmarks(frame)
        all_landmarks.append(landmarks)
    cap.release()
    return extract_embedding(all_landmarks, model)


st.title("Real-Time Video Matching")


# Columns
col1, col2 = st.columns(2)

with col1:
    st.header("Video Tham Chiếu")
    ref_video = cv2.VideoCapture(VIDEO_PATH)
    ref_landmarks = []

    while ref_video.isOpened():
        ret, frame = ref_video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ref_landmarks.append(get_frame_landmarks(frame))
        st.image(frame, channels="RGB", use_column_width=True)

    ref_video.release()
    reference_embedding = extract_embedding(ref_landmarks, model)

with col2:
    st.header("Camera Realtime")
    cap = cv2.VideoCapture(0)

    all_landmarks = []
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = get_frame_landmarks(frame_rgb)
        all_landmarks.append(landmarks)
        frame_counter += 1

        if frame_counter == 60:
            current_embedding = extract_embedding(all_landmarks, model)
            similarity = calculate_cosine_similarity(current_embedding, reference_embedding)

            if similarity > THRESHOLD:
                st.write(f"Similarity: {similarity:.2f} - Matched")
            else:
                st.write(f"Similarity: {similarity:.2f} - Not Matched")

            frame_counter = 0
            all_landmarks = []

        st.image(frame, channels="RGB", use_column_width=True)

    cap.release()
