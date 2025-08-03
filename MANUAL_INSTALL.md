# Manual Installation Guide

Step-by-step manual installation instructions for Qwen3-Coder-480B-A35B-Instruct.

## 📋 Prerequisites

Before starting, ensure your system meets the requirements:

- **OS**: Ubuntu 20.04+ or 22.04 LTS
- **GPU**: NVIDIA H100 80GB or A100 80GB (minimum)
- **RAM**: 64GB+ system memory
- **Storage**: 500GB+ free space
- **Network**: High-speed internet connection

## 🛠️ Step 1: System Preparation

### Update System Packages

```bash
sudo apt update && sudo apt upgrade -y
```

### Install Essential Dependencies

```bash
sudo apt install -y \
    build-essential \
    cmake \
    curl \
    wget \
    git \
    git-lfs \
    htop \
    nvtop \
    tmux \
    vim \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    liblzma-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    bc
```

### Set up Git LFS

```bash
git lfs install --skip-repo
```

## 🎮 Step 2: NVIDIA GPU Setup

### Verify GPU Driver

```bash
nvidia-smi
```

If not working, install NVIDIA drivers:

```bash
sudo apt install nvidia-driver-535
sudo reboot
```

### Install CUDA 12.1

```bash
# Download CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update

# Install CUDA toolkit
sudo apt install -y cuda-toolkit-12-1

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
```

## 🐍 Step 3: Python Environment Setup

### Create Virtual Environment

```bash
# Define installation directory
export INSTALL_DIR="$HOME/qwen480b_env"

# Remove existing installation if present
rm -rf "$INSTALL_DIR"

# Create new virtual environment
python3 -m venv "$INSTALL_DIR"

# Activate environment
source "$INSTALL_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Install PyTorch with CUDA Support

```bash
# Install PyTorch 2.3.0 with CUDA 12.1
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

### Install Core Dependencies

```bash
pip install \
    transformers==4.54.1 \
    accelerate==0.33.0 \
    tokenizers==0.19.1 \
    sentencepiece==0.2.0 \
    protobuf==3.20.3 \
    huggingface-hub==0.24.6 \
    peft==0.12.0 \
    bitsandbytes==0.43.3 \
    datasets==2.21.0 \
    evaluate==0.4.3 \
    scikit-learn==1.5.1 \
    scipy==1.13.1 \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    jupyter==1.0.0 \
    ipython==8.26.0 \
    tqdm==4.66.5 \
    psutil==6.0.0 \
    gpustat==1.1.1 \
    nvidia-ml-py3==7.352.0
```

### Install Optional Optimization Libraries

```bash
# VLLM for optimized inference (may fail on some systems)
pip install vllm==0.5.4

# Flash Attention (if compatible)
pip install flash-attn --no-build-isolation
```

## 📦 Step 4: Model Download

### Set Up Hugging Face Environment

```bash
# Set up cache directory
export HF_HOME="$INSTALL_DIR/huggingface_cache"
mkdir -p "$HF_HOME"

# Create model directory
mkdir -p "$INSTALL_DIR/models"
cd "$INSTALL_DIR/models"
```

### Download Model (Option 1: Python API)

```bash
python3 << 'EOF'
import os
from huggingface_hub import snapshot_download

model_repo = "Qwen/Qwen2.5-Coder-32B-Instruct"
model_dir = os.path.join(os.environ['INSTALL_DIR'], 'models', 'qwen3-coder-480b')

print(f"Downloading {model_repo}...")
print("This will download approximately 450GB of data...")

snapshot_download(
    repo_id=model_repo,
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    token=os.environ.get('HF_TOKEN'),  # Add token if needed
)

print(f"Model downloaded to: {model_dir}")
EOF
```

### Download Model (Option 2: Git Clone)

```bash
# Alternative method using git
cd "$INSTALL_DIR/models"
git clone https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct qwen3-coder-480b
cd qwen3-coder-480b
git lfs pull
```

## 🧪 Step 5: Verification

### Create Test Script

```bash
cat > "$INSTALL_DIR/test_setup.py" << 'EOF'
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def test_installation():
    print("🧪 Testing Qwen3-Coder-480B Installation")
    print("=" * 50)
    
    # Configuration
    model_dir = os.path.join(os.environ['INSTALL_DIR'], 'models', 'qwen3-coder-480b')
    
    try:
        # Load tokenizer
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        print("✓ Tokenizer loaded successfully")
        
        # Load model
        print("🧠 Loading model...")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f}s")
        
        # Check GPU memory
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"💾 GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB")
        
        # Test inference
        print("🎯 Testing inference...")
        prompt = "def fibonacci(n):"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        inference_time = time.time() - start_time
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✓ Inference completed in {inference_time:.2f}s")
        print(f"📝 Response preview: {response[:200]}...")
        
        print("\n🎉 Installation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_installation()
    exit(0 if success else 1)
EOF
```

### Run Verification Test

```bash
# Set environment variables
export INSTALL_DIR="$HOME/qwen480b_env"
source "$INSTALL_DIR/bin/activate"

# Run test
python "$INSTALL_DIR/test_setup.py"
```

## 🚀 Step 6: Create Activation Script

```bash
cat > "$INSTALL_DIR/activate_qwen480b.sh" << 'EOF'
#!/bin/bash
# Qwen3-Coder-480B Environment Activation Script

echo "🚀 Activating Qwen3-Coder-480B Environment"

# Activate Python environment
source "$HOME/qwen480b_env/bin/activate"

# Set environment variables
export INSTALL_DIR="$HOME/qwen480b_env"
export HF_HOME="$INSTALL_DIR/huggingface_cache"
export CUDA_VISIBLE_DEVICES=0

# Add CUDA to PATH
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

echo "✓ Environment activated!"
echo "Python: $(which python)"
echo "Install dir: $INSTALL_DIR"

# Change to install directory
cd "$INSTALL_DIR"
EOF

chmod +x "$INSTALL_DIR/activate_qwen480b.sh"

# Create symlink for easy access
ln -sf "$INSTALL_DIR/activate_qwen480b.sh" "$HOME/activate_qwen480b.sh"
```

## 📝 Step 7: Create Usage Examples

### Basic Inference Script

```bash
cat > "$INSTALL_DIR/basic_inference.py" << 'EOF'
#!/usr/bin/env python3
"""Basic inference example for Qwen3-Coder-480B"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Load model
model_dir = os.path.join(os.environ['INSTALL_DIR'], 'models', 'qwen3-coder-480b')
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Example usage
prompt = "Write a Python function to calculate prime numbers:"
inputs = tokenizer(prompt, return_tensors="pt")

if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
EOF

chmod +x "$INSTALL_DIR/basic_inference.py"
```

## ✅ Step 8: Final Verification

### Quick System Check

```bash
# Activate environment
source ~/activate_qwen480b.sh

# Check installation
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Test model loading (quick check)
python -c "
from transformers import AutoTokenizer
import os
model_dir = os.path.join(os.environ['INSTALL_DIR'], 'models', 'qwen3-coder-480b')
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
print('✓ Model files accessible')
"

# Check GPU memory
nvidia-smi
```

## 📚 Next Steps

1. **Run Full Test**: `python basic_inference.py`
2. **Monitor Performance**: `watch -n 1 nvidia-smi`
3. **Read Documentation**: Check `docs/` directory
4. **Join Community**: [GitHub Discussions](https://github.com/twobitapps/480b-setup/discussions)

## 🐛 Troubleshooting

If you encounter issues:

1. **Check logs**: Look for error messages in terminal output
2. **Verify GPU**: Run `nvidia-smi` to ensure GPU is available
3. **Check memory**: Ensure sufficient GPU and system memory
4. **Review dependencies**: Verify all packages installed correctly
5. **Consult troubleshooting guide**: See `docs/TROUBLESHOOTING.md`

## 🔧 Advanced Configuration

### Environment Variables

Add to `~/.bashrc` for persistent configuration:

```bash
# Qwen 480B Configuration
export INSTALL_DIR="$HOME/qwen480b_env"
export HF_HOME="$INSTALL_DIR/huggingface_cache"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
```

### Memory Optimization

For systems with limited memory:

```python
# Use model sharding
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    max_memory={0: "40GiB"},  # Limit GPU memory usage
    offload_folder="./offload",
    torch_dtype=torch.float16,
)
```

### Performance Tuning

```python
# Enable optimizations
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use torch.compile for faster inference (PyTorch 2.0+)
model = torch.compile(model, mode="reduce-overhead")
```

## 📊 Expected Performance

On NVIDIA H100 80GB:
- **Model Loading**: 2-3 minutes
- **First Inference**: 10-15 seconds
- **Subsequent Inference**: 3-6 seconds
- **Tokens per Second**: 100-200 (varies by complexity)
- **GPU Memory Usage**: ~45-50GB

---

**🎉 Congratulations!** You have successfully installed Qwen3-Coder-480B-A35B-Instruct manually. The model is now ready for use.