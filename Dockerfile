# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and other packages
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application and sample data into the container
COPY streamlit_app.py ./
COPY sample_data/ sample_data/

# Expose the port Streamlit will run on
EXPOSE 8501

# Default command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py"]
