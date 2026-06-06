# Use lightweight Python image
FROM python:3.10-slim

# Install system dependencies for OpenCV / YOLOv8 (required for cv2 / ultralytics)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (leverage Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Hugging Face Spaces defaults to port 7860
EXPOSE 7860

# Use gunicorn as production WSGI server
# $PORT is set by Hugging Face Spaces to 7860
# --timeout 300 gives the model 5 minutes to load on first request
# --workers 1 keeps memory usage low for the ML model
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1
