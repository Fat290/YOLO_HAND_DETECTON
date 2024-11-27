import cv2
import numpy as np
from keras.models import load_model
import torch
from pathlib import Path
import src.configs as cf
import time
import pathlib

# Fix for Windows Path issue
pathlib.PosixPath = pathlib.WindowsPath

# Paths to YOLOv5 and Keras models
yolo_weights_path = str(Path('./yolov5/runs/train/exp/weights/best.pt'))
keras_model_path = str(Path('./model/fine_tune_asl_model.h5'))

# Load YOLOv5 model
try:
    yolo_model = torch.hub.load(
        './yolov5',
        'custom',
        path=yolo_weights_path,
        source='local',
        force_reload=True
    )
    print("YOLOv5 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    exit(1)

# Load Keras model
try:
    sign_model = load_model(keras_model_path)
    print("Keras model loaded successfully.")
except Exception as e:
    print(f"Error loading Keras model: {e}")
    exit(1)

# Function to wrap text into multiple lines
def wrap_text(text, max_line_length=20):
    """
    Wrap text into multiple lines based on the max_line_length.
    """
    lines = []
    while len(text) > max_line_length:
        space_index = text.rfind(' ', 0, max_line_length)
        if space_index == -1:  # No space found, force break
            space_index = max_line_length
        lines.append(text[:space_index])
        text = text[space_index:].strip()
    if text:
        lines.append(text)  # Add the remaining text
    return lines

def recognize():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 520)

    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    text, word = "", ""
    count_same_frame = 0
    padding = 80
    frame_count = 0  # Add frame count variable

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        frame = cv2.flip(frame, 1)  # Flip frame horizontally
        start = time.time()

        # Skip 4 frames per second (process every 5th frame)
        frame_count += 1
        if frame_count % 5 != 0:  # Skip frame if it's not every 5th frame
            continue

        try:
            results = yolo_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = results.pandas().xyxy[0]
        except Exception as e:
            print(f"Error during YOLOv5 detection: {e}")
            break

        for _, row in detections.iterrows():
            if row['name'] == 'hand' and row['confidence'] > 0.5:
                xmin = max(0, int(row['xmin']) - padding)
                ymin = max(0, int(row['ymin']) - padding)
                xmax = min(frame.shape[1], int(row['xmax']) + padding)
                ymax = min(frame.shape[0], int(row['ymax']) + padding)

                cropped_hand = frame[ymin:ymax, xmin:xmax]

                try:
                    resized_frame = cv2.resize(cropped_hand, (cf.IMAGE_SIZE, cf.IMAGE_SIZE))
                    reshaped_frame = np.array(resized_frame).reshape((1, cf.IMAGE_SIZE, cf.IMAGE_SIZE, 3))
                    frame_for_model = reshaped_frame / 255.0

                    old_text = text
                    prediction = sign_model.predict(frame_for_model)
                    prediction_probability = prediction[0, prediction.argmax()]
                    text = cf.CLASSES[prediction.argmax()]
                except Exception as e:
                    print(f"Error during Keras model prediction: {e}")
                    prediction_probability = 0.0  # Default value
                    text = "nothing"  # Default value
                    continue

                if text == 'space':
                    text = '_'
                if text != 'nothing':
                    if old_text == text:
                        count_same_frame += 1
                    else:
                        count_same_frame = 0

                    if count_same_frame > 4:
                        word += text
                        count_same_frame = 0

                if prediction_probability > 0.7:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f"{text} ({prediction_probability * 100:.2f}%)",
                                (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Blackboard for displaying predictions
        blackboard = np.zeros((350, 350, 3), dtype=np.uint8)
        
        # Wrap text for display to avoid overflow
        wrapped_text = wrap_text(f"Prediction: {text}", max_line_length=12)
        wrapped_word = wrap_text(f"Word: {word}", max_line_length=10)
        
        # Display wrapped text on the blackboard
        y_offset = 100  # Starting y-coordinate
        for line in wrapped_text:
            cv2.putText(blackboard, line, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            y_offset += 40  # Increase offset for next line

        if 'prediction_probability' in locals() and prediction_probability > 0:  # Check existence
            wrapped_probability = wrap_text(f"Probability: {prediction_probability * 100:.2f}%", max_line_length=20)
            for line in wrapped_probability:
                cv2.putText(blackboard, line, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                y_offset += 40  

        for line in wrapped_word:
            cv2.putText(blackboard, line, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            y_offset += 40  

        
        blackboard_resized = cv2.resize(blackboard, (blackboard.shape[1], frame.shape[0]))

       
        combined_frame = np.hstack((frame, blackboard_resized))

        cv2.imshow("Hand Detection & Sign Language Recognition", combined_frame)

        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):  # Quit
            break
        if k == ord('r'):  # Reset word
            word = ""
        if k == ord('z'):  # Remove last character
            word = word[:-1]

    cam.release()
    cv2.destroyAllWindows()

recognize()
