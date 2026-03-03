FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by opencv, python packages, etc.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables for credentials
ENV KAGGLE_CONFIG_DIR=/root/.kaggle

# Expose ports for both APIs
EXPOSE 8000
EXPOSE 8001

# Default command
CMD ["bash"]
