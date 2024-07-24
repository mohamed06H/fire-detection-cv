import streamlit as st
from ultralytics import YOLO
import cvzone
import cv2
import math
import tempfile
import mlflow
import os

# Set the MLflow tracking URI to the remote server
mlflow.set_tracking_uri("https://mlflow-server.duckdns.org/")


# Function to load the YOLO model from MLflow
def load_model(model_name, stage='Production'):
    # Get the model URI
    model_uri = f"models:/{model_name}/{stage}"

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the specific artifact from MLflow
        artifact_path = 'fire_yolo.pt'
        model_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=temp_dir)

        # Construct the full path to the downloaded model
        full_model_path = os.path.join(model_path, artifact_path)

        # Ensure the file exists before trying to load it
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"Model file not found at {full_model_path}")

        # Load the model using Ultralytics
        model = YOLO(full_model_path)

        return model


# Load the YOLO model
model_name = 'fire_detection_yolo'
model = load_model(model_name)


# Title of the Streamlit app
st.title("Fire Detection using YOLO")

# Upload video file
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

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

        # Getting bbox, confidence and class names information to work with
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
