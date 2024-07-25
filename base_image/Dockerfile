# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /base

# Install system dependencies required for OpenCV and other packages in one RUN statement
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY ../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# This image now contains all the necessary system and Python dependencies
