#!/bin/bash

echo "🚀 Starting Qwen3-Coder-480B Container"

# Activate Python environment
source /opt/qwen480b_env/bin/activate

# Check if model exists, if not provide instructions
if [ ! -d "/opt/qwen480b_env/models/qwen3-coder-480b" ]; then
    echo "⚠️  Model not found. Please download the model first:"
    echo "  docker exec -it qwen480b-setup /app/scripts/model_download.sh"
fi

# Start interactive shell or run command
if [ "$#" -eq 0 ]; then
    echo "🐚 Starting interactive shell"
    exec /bin/bash
else
    echo "🏃 Running command: $@"
    exec "$@"
fi