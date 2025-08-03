# Qwen3-Coder-480B-A35B-Instruct Docker Image
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    git-lfs \
    curl \
    wget \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set up Git LFS
RUN git lfs install --skip-repo

# Create virtual environment
RUN python3 -m venv /opt/qwen480b_env

# Install Python dependencies
COPY config/requirements.txt /tmp/requirements.txt
RUN /opt/qwen480b_env/bin/pip install --upgrade pip && \
    /opt/qwen480b_env/bin/pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    /opt/qwen480b_env/bin/pip install -r /tmp/requirements.txt

# Set environment variables
ENV INSTALL_DIR=/opt/qwen480b_env
ENV HF_HOME=/opt/qwen480b_env/huggingface_cache
ENV PATH="/opt/qwen480b_env/bin:$PATH"

# Copy scripts and examples
COPY scripts/ /app/scripts/
COPY examples/ /app/examples/
COPY config/ /app/config/

# Create model directory
RUN mkdir -p /opt/qwen480b_env/models

# Set entry point
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]