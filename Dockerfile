# Use the custom base image from the Docker registry
FROM mohamed06/wf_base_image:latest

# Set working directory
WORKDIR /app

# Copy the application and sample data into the container
COPY streamlit_app.py ./
COPY sample_data/ sample_data/

# Expose the port Streamlit will run on
EXPOSE 8501

# Default command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py"]
