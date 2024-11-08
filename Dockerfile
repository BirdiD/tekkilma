# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# CUDA and GPU settings
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# HuggingFace cache settings
ENV HF_HOME="/cache/huggingface"
ENV HF_DATASETS_CACHE="/cache/huggingface/datasets"
ENV DEFAULT_HF_METRICS_CACHE="/cache/huggingface/metrics"
ENV DEFAULT_HF_MODULES_CACHE="/cache/huggingface/modules"
ENV HUGGINFACE_HUB_CACHE="/cache/huggingface/hub"
ENV HUGGINGFACE_ASSETS_CACHE="/cache/huggingface/assets"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Create cache directories
RUN mkdir -p /cache/huggingface/hub/tekkilma-24000

# Install Python Dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install --no-cache-dir -r /requirements.txt && \
    rm /requirements.txt


# Cache Models with GPU support
COPY builder/cache_model.py /cache_model.py
RUN python3.10 /cache_model.py && \
    rm /cache_model.py


# Copy source code
COPY src .

# Basic validation
RUN test -n "$(ls -A /cache/huggingface)" || exit 1

CMD ["python", "-u", "handler.py"]