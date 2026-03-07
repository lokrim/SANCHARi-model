FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python and system dependencies in one layer, clean up after
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PyTorch with CUDA 11.8
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install remaining packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories that may be overridden by volume mounts at runtime.
RUN mkdir -p /app/predicted/predicted \
             /app/predicted/output-geojson \
             /app/predicted/predictedv4 \
             /app/geotiffs \
             /app/weights \
             /app/data

ENV KAGGLE_CONFIG_DIR=/root/.kaggle
ENV PYTHONPATH=/app/src
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000
EXPOSE 8001

CMD ["bash"]