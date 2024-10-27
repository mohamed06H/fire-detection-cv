# Fire Detection using YOLOv8
This repository contains the results of our experiment on Aerial images fire detection, a report is available under /best_experiment.

[Link to study and datasets](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)


This repository also contains a Streamlit application that performs fire detection on videos using a pre-trained YOLOv8 model. The application can load the model from a local file and perform inference on sample or user-uploaded videos.

For the aim of simplifying things, the best model was downloaded manually from MLFlow to /models/best_model_fire.pt, more automation with CI/CD was done in (amazon_linux) branch.

## Table of Contents
- [Fire Detection using YOLOv8](#fire-detection-using-yolov8)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Clone the Repository](#clone-the-repository)
    - [Install Dependencies](#install-dependencies)
    - [Folder Structure](#folder-structure)
  - [Usage](#usage)
    - [Run Locally](#run-locally)
    - [Load Model](#load-model)
  - [Docker Usage](#docker-usage)
    - [Build Docker Image](#build-docker-image)
    - [Run Docker Container](#run-docker-container)
    - [Application Structure](#application-structure)
    - [Model Information](#model-information)
  - [Contributing](#contributing)
    - [Steps to Contribute](#steps-to-contribute)

## Features

- Fire detection using YOLOv8 model.
- Real-time video processing and detection.
- Support for sample videos or user-uploaded video files.
- Local file-based model loading for inference.

## Installation

### Prerequisites

- **Python 3.10** or higher.
- **pip** (Python package installer)
- **Docker** (Optional, for containerized deployment)

### Clone the Repository

```bash
git clone https://github.com/yourusername/fire-detection-yolov8.git
cd fire-detection-yolov8
```

### Install Dependencies
Make sure to install the necessary Python packages:

```bash
pip install -r requirements.txt
```

### Folder Structure
```graphql
fire-detection-yolov8/
├── models/
│   └── fire_yolo.pt           # Pre-trained YOLO model file
├── sample_data/
│   ├── fire1.mp4              # Sample video file 1
│   └── fire2.mp4              # Sample video file 2
├── best_experiment/           # Dockerfile for containerizing the app
│   ├── azureml_run_dl_...py   # Experiment notebook with both train and testing phases
│   └── save_at_36.h5          # best model downloaded from AzureML
├── streamlit_app.py           # Main Streamlit application
├── Dockerfile                 # Dockerfile for containerizing the app
├── fire_detection_yolo.ipyng  # Collab notebook for fine-tunning the pre-trained yolo model
├── project_report.docx        # Experiment report on Tensorflow and AzureML
├── requirements.txt           # Python dependencies
└── README.md                  # This README file
```

## Usage
### Run Locally
To start the Streamlit application locally, navigate to the project directory and run:

```bash
streamlit run streamlit_app.py
```
The application will be accessible at http://localhost:8501.

### Load Model
Ensure that the YOLO model (fire_yolo.pt) is in the models/ folder. Modify the path in streamlit_app.py if necessary:

```python
local_model_path = 'models/fire_yolo.pt'
```

## Docker Usage
### Build Docker Image
To build the Docker image for the Streamlit app, use the following command:

```bash
docker build -t fire-detection-app .
```
It will take a few minutes for installing dependencies during the first stage. 

Multi-Stage build is used to accelerate next builds.

### Run Docker Container
Once the image is built, you can run the container:

```bash
docker run -p 8501:8501 fire-detection-app
```
The application will be accessible at http://localhost:8501.

### Application Structure
- streamlit_app.py: The main file containing the Streamlit application logic.
- models/: Directory containing the pre-trained YOLO model (fire_yolo.pt).
- sample_data/: Directory containing sample video files for testing.

### Model Information
The model used in this project is a pre-trained YOLOv8 model, specifically trained for fire detection. Ensure the model file (fire_yolo.pt) is placed in the models/ directory.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue if you encounter any problems.

### Steps to Contribute
- Fork the repository.
- Create a new branch: git checkout -b feature-name
- Make your changes.
- Commit your changes: git commit -m 'Add some feature'
- Push to the branch: git push origin feature-name
- Submit a pull request.

