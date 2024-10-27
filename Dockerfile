# Stage 1: Build the base image with dependencies
FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory for the base image
WORKDIR /base

# Install system dependencies required for OpenCV, Git, and other packages in one RUN statement
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file before other files to leverage Docker cache
COPY requirements.txt ./ 

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Build the final image with the application
FROM base AS final

# Set working directory for the application
WORKDIR /app

# Copy only the application files and model files to avoid reinstalling dependencies
COPY streamlit_app.py ./ 
COPY sample_data/ sample_data/
COPY models/ models/

# Expose the port Streamlit will run on
EXPOSE 8501

# Default command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py"]
