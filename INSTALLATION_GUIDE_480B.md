# Qwen3-Coder-480B Installation Guide (Verified Working)

This guide provides the exact steps that were successfully used to install and run the Qwen3-Coder-480B model locally.

## 📋 Prerequisites

### Hardware Requirements (Verified)
- **GPU**: 4x NVIDIA H100 80GB (or similar high-memory GPUs)
- **RAM**: 64GB+ system RAM (885GB used in test system)
- **Storage**: 300GB+ free space for quantized model
- **CPU**: Multi-core processor (104 cores in test system)

### Software Requirements
- **OS**: Ubuntu 22.04 LTS
- **CUDA**: 12.1+ (12.8 used in test)
- **Python**: 3.8-3.11
- **CMake**: Required for building llama.cpp

## 🚀 Step-by-Step Installation

### Step 1: Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y \
    build-essential \
    cmake \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-venv \
    bc

# Verify CUDA installation
nvidia-smi
```

### Step 2: Set Up Python Environment

```bash
# Create and activate virtual environment
python3 -m venv ~/qwen480b_env
source ~/qwen480b_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Hugging Face Hub CLI
pip install huggingface-hub
```

### Step 3: Download the Quantized Model

```bash
# Create model directory
mkdir -p ~/models/qwen3-coder-480b-q4km
cd ~/models/qwen3-coder-480b-q4km

# Download Q4_K_M quantized version (~270GB total)
# This downloads 6 GGUF files
hf download unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF \
  --include="Q4_K_M/*" \
  --local-dir .

# Download README (optional)
hf download unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF \
  --include="README.md" \
  --local-dir .
```

### Step 4: Build llama.cpp with CUDA Support

```bash
# Clone llama.cpp
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Create build directory
mkdir -p build && cd build

# Configure with CMake (disable CURL to avoid dependency issues)
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF

# Build (uses all CPU cores)
make -j$(nproc)
```

### Step 5: Create Run Script

```bash
# Create the run script
cat > ~/run_qwen3_480b.sh << 'EOF'
#!/bin/bash
# Qwen3-Coder-480B Inference Script

MODEL_DIR="$HOME/models/qwen3-coder-480b-q4km"
LLAMA_CPP="$HOME/llama.cpp"

echo "🚀 Starting Qwen3-Coder-480B inference server..."
echo "Model directory: $MODEL_DIR"
echo "LLAMA.CPP: $LLAMA_CPP"

# Check if model files exist
if [ ! -d "$MODEL_DIR/Q4_K_M" ]; then
    echo "❌ Model files not found in $MODEL_DIR/Q4_K_M"
    exit 1
fi

# Use the first GGUF file as main model
MAIN_MODEL=$(find "$MODEL_DIR/Q4_K_M" -name "*-00001-of-*.gguf" | head -1)

echo "Using main model file: $MAIN_MODEL"

# Run llama.cpp server with multi-GPU support
"$LLAMA_CPP/build/bin/llama-server" \
    --model "$MAIN_MODEL" \
    --ctx-size 8192 \
    --n-gpu-layers 99 \
    --tensor-split 1,1,1,1 \
    --host 0.0.0.0 \
    --port 8080 \
    --threads $(nproc) \
    --cont-batching \
    --mlock \
    --verbose
EOF

# Make it executable
chmod +x ~/run_qwen3_480b.sh
```

## 🧪 Verification

### Test with Command Line Interface

```bash
cd ~/llama.cpp/build

./bin/llama-cli \
  --model ~/models/qwen3-coder-480b-q4km/Q4_K_M/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00006.gguf \
  --prompt "Write a Python function to calculate fibonacci:" \
  --n-predict 100 \
  --temp 0.7
```

### Start the Server

```bash
# Start the server
~/run_qwen3_480b.sh

# In another terminal, test the API
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function to calculate fibonacci:",
    "n_predict": 100,
    "temperature": 0.7
  }'
```

## 📊 Expected Output

When successful, you should see:
- Model info showing "Qwen3-Coder-480B-A35B-Instruct" with 480.15B parameters
- All 4 GPUs detected and utilized
- Server listening on port 8080
- Intelligent code generation responses

### Performance Metrics (on 4x H100 system)
- **Model Loading**: ~14.5 seconds
- **Inference Speed**: ~7.16 tokens/second
- **GPU Memory Usage**: Distributed across all GPUs

## 🔧 Alternative Quantizations

The GGUF repository offers various quantization levels:

| Quantization | Size | Quality | Use Case |
|--------------|------|---------|----------|
| Q2_K | ~90GB | Lower | Fast inference, lower quality |
| Q4_K_M | ~270GB | Good | Balanced quality/size (recommended) |
| Q5_K_M | ~315GB | Better | Higher quality, more resources |
| Q8_0 | ~480GB | Best | Highest quality, maximum resources |

To use a different quantization, modify the download command:
```bash
# Example for Q5_K_M
hf download unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF \
  --include="Q5_K_M/*" \
  --local-dir .
```

## 🐛 Troubleshooting

### Issue: CMake error about CURL
**Solution**: Add `-DLLAMA_CURL=OFF` to cmake command

### Issue: Server won't start
**Solution**: Check that all 6 GGUF files downloaded correctly

### Issue: Out of memory
**Solution**: Use a smaller quantization (Q2_K or Q3_K_M)

### Issue: Slow inference
**Solution**: Ensure `--n-gpu-layers 99` is set to offload to GPU

## 📝 Notes

- The model files are split into 6 parts but llama.cpp loads them automatically
- The `--tensor-split 1,1,1,1` parameter distributes the model evenly across 4 GPUs
- Context size can be increased up to the model's native 262,144 tokens
- The Q4_K_M quantization provides a good balance of quality and resource usage

## 🎉 Success Indicators

You'll know the installation is successful when:
1. ✅ All 6 GGUF files are downloaded (~270GB total)
2. ✅ llama.cpp builds without errors
3. ✅ The CLI test generates coherent code
4. ✅ The server starts and responds to API requests
5. ✅ GPU utilization is visible in `nvidia-smi`

---

**Verified Working**: This guide was tested on Ubuntu 22.04 with 4x NVIDIA H100 80GB GPUs and successfully runs the Qwen3-Coder-480B model.