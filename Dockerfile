# Use NVIDIA PyTorch base image with CUDA 11.8 support
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace/LaBraM

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tensorboardX && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files (excluding large files - see .dockerignore)
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints log datasets

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV CUDA_VISIBLE_DEVICES=0

# Expose tensorboard port
EXPOSE 6006

# Default command (can be overridden)
CMD ["/bin/bash"]