import streamlit as st
from ultralytics import YOLO
import cvzone
import cv2
import math
import os
import tempfile

# Function to load the YOLO model from a local file
# Lazy-load the YOLO model
@st.cache_resource
def load_model(local_model_path):
    # Ensure the file exists before trying to load it
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Model file not found at {local_model_path}")

    # Load the model using Ultralytics
    model = YOLO(local_model_path)

    return model

# Path to the local YOLO model file
local_model_path = 'models/fire_yolo.pt'

# Load the YOLO model
model = load_model(local_model_path)

# Title of the Streamlit app
st.title("Fire Detection using YOLOv8")

# Option for selecting between sample files or uploading a file
option = st.radio("Choose video input method:", ("Sample Files", "Upload Your Own"))

temp_file_path = None

if option == "Sample Files":
    # Sample files to choose from
    sample_files = ["fire1.mp4", "fire2.mp4"]

    if sample_files:
        # Select a sample video file
        selected_file = st.selectbox("Select a sample video file", sample_files,
                                     index=None)

        if selected_file:
            # Load selected sample file
            sample_file_path = os.path.join("sample_data", selected_file)
            if not os.path.isfile(sample_file_path):
                st.error("Sample file not found.")
            else:
                temp_file_path = sample_file_path
    else:
        st.error("No sample files available.")

elif option == "Upload Your Own":
    # Upload video file
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

# Run inference if a file path is set
if temp_file_path:
    # Reading the classes
    classnames = ['fire']

    # Open video file
    cap = cv2.VideoCapture(temp_file_path)

    # Create a placeholder for the video
    video_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        result = model(frame, stream=True)

        # Getting bbox, confidence, and class names information to work with
        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 50:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                       scale=1.5, thickness=2)

        # Display the video frame
        video_placeholder.image(frame, channels="BGR")

    cap.release()
