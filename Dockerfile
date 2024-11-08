# Base image
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV HF_HOME="/cache/huggingface"
ENV HF_DATASETS_CACHE="/cache/huggingface/datasets"
ENV DEFAULT_HF_METRICS_CACHE="/cache/huggingface/metrics"
ENV DEFAULT_HF_MODULES_CACHE="/cache/huggingface/modules"
ENV HUGGINFACE_HUB_CACHE="/cache/huggingface/hub"
ENV HUGGINGFACE_ASSETS_CACHE="/cache/huggingface/assets"

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /workspace

# install system package ffmpeg
RUN apt-get update && apt-get install -y ffmpeg


RUN mkdir -p /cache/huggingface/hub/tekkilma-24000

# Disable CUDA during build
ENV CUDA_VISIBLE_DEVICES=""

# Install Python Dependencies
COPY builder/requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt && \
    rm /requirements.txt

# Cache Models
COPY builder/cache_model.py /cache_model.py
ENV CUDA_VISIBLE_DEVICES=""  # Disable CUDA during build
RUN python /cache_model.py && \
    rm /cache_model.py

# Copy Source Code
ADD src .

# Basic validation
# Verify that the cache folder is not empty
RUN test -n "$(ls -A /cache/huggingface)"

# Enable CUDA for runtime
ENV CUDA_VISIBLE_DEVICES="all"
ENV NVIDIA_VISIBLE_DEVICES="all"

CMD ["python", "-u", "handler.py"]