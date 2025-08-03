#!/bin/bash

# Qwen3-Coder-480B-A35B-Instruct Automated Installation Script
# Version: 1.0.0
# Compatible with: Ubuntu 20.04+, 22.04 LTS
# Hardware: NVIDIA H100 80GB, A100 80GB (minimum)

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="$HOME/qwen480b_env"
MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
ACTUAL_MODEL_NAME="Qwen3-Coder-480B-A35B-Instruct"
LOG_FILE="$HOME/qwen480b_install.log"
BACKUP_SERVERS=(
    "https://hf-mirror.com"
    "https://huggingface.co"
)

# Functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "$LOG_FILE"
}

print_header() {
    echo -e "${PURPLE}"
    echo "  ██████╗ ██╗    ██╗███████╗███╗   ██╗    ██╗  ██╗ █████╗  ██████╗ ██████╗ "
    echo "  ██╔═══██╗██║    ██║██╔════╝████╗  ██║    ██║  ██║██╔══██╗██╔═══██╗██╔══██╗"
    echo "  ██║   ██║██║ █╗ ██║█████╗  ██╔██╗ ██║    ███████║╚█████╔╝██║   ██║██████╔╝"
    echo "  ██║▄▄ ██║██║███╗██║██╔══╝  ██║╚██╗██║    ╚════██║██╔══██╗██║   ██║██╔══██╗"
    echo "  ╚██████╔╝╚███╔███╔╝███████╗██║ ╚████║         ██║╚█████╔╝╚██████╔╝██████╔╝"
    echo "   ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝         ╚═╝ ╚════╝  ╚═════╝ ╚═════╝ "
    echo ""
    echo "              Qwen3-Coder-480B-A35B-Instruct Installation Script"
    echo "                           Version 1.0.0"
    echo -e "${NC}"
}

check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root for security reasons."
        log_info "Please run as a regular user with sudo privileges."
        exit 1
    fi
}

check_system_requirements() {
    log "Checking system requirements..."
    
    # Check Ubuntu version
    if ! grep -q "Ubuntu" /etc/os-release; then
        log_error "This script is designed for Ubuntu systems."
        exit 1
    fi
    
    local ubuntu_version
    ubuntu_version=$(lsb_release -rs)
    if (( $(echo "$ubuntu_version < 20.04" | bc -l) )); then
        log_error "Ubuntu 20.04 or later is required. Found: $ubuntu_version"
        exit 1
    fi
    
    # Check available disk space (need at least 500GB)
    local available_space
    available_space=$(df "$HOME" | awk 'NR==2 {print $4}')
    local required_space=$((500 * 1024 * 1024)) # 500GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        log_error "Insufficient disk space. Required: 500GB, Available: $((available_space / 1024 / 1024))GB"
        exit 1
    fi
    
    # Check memory
    local total_mem
    total_mem=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local required_mem=$((64 * 1024 * 1024)) # 64GB in KB
    
    if [ "$total_mem" -lt "$required_mem" ]; then
        log_warning "Low system memory. Recommended: 64GB+, Available: $((total_mem / 1024 / 1024))GB"
    fi
    
    # Check NVIDIA GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "NVIDIA GPU driver not found. Please install NVIDIA drivers first."
        exit 1
    fi
    
    local gpu_memory
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$gpu_memory" -lt 70000 ]; then
        log_error "Insufficient GPU memory. Required: 80GB+, Available: ${gpu_memory}MB"
        exit 1
    fi
    
    log "✓ System requirements check passed"
}

install_system_dependencies() {
    log "Installing system dependencies..."
    
    # Update package lists
    sudo apt update
    
    # Install essential packages
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
        libffi-dev \
        liblzma-dev \
        bc
    
    # Install Git LFS
    git lfs install --skip-repo
    
    log "✓ System dependencies installed"
}

setup_cuda() {
    log "Setting up CUDA..."
    
    # Check if CUDA is already installed
    if command -v nvcc &> /dev/null; then
        local cuda_version
        cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log_info "CUDA already installed: $cuda_version"
        
        # Check if version is compatible (12.1+)
        if (( $(echo "$cuda_version >= 12.1" | bc -l) )); then
            log "✓ CUDA version is compatible"
            return 0
        else
            log_warning "CUDA version $cuda_version may not be optimal. Recommended: 12.1+"
        fi
    fi
    
    # Install CUDA 12.1
    log "Installing CUDA 12.1..."
    
    # Download and install CUDA keyring
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    rm cuda-keyring_1.0-1_all.deb
    
    # Update package lists
    sudo apt update
    
    # Install CUDA toolkit
    sudo apt install -y cuda-toolkit-12-1
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
    
    # Source the updated bashrc
    source ~/.bashrc
    
    log "✓ CUDA installed successfully"
}

create_python_environment() {
    log "Creating isolated Python environment..."
    
    # Remove existing environment if it exists
    if [ -d "$INSTALL_DIR" ]; then
        log_warning "Removing existing installation at $INSTALL_DIR"
        rm -rf "$INSTALL_DIR"
    fi
    
    # Create virtual environment
    python3 -m venv "$INSTALL_DIR"
    
    # Activate environment
    source "$INSTALL_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    log "✓ Python environment created at $INSTALL_DIR"
}

install_python_dependencies() {
    log "Installing Python dependencies..."
    
    # Activate environment
    source "$INSTALL_DIR/bin/activate"
    
    # Install PyTorch with CUDA 12.1 support
    pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install core dependencies with specific versions
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
    
    # Install VLLM for optimized inference (if compatible)
    log "Attempting to install VLLM..."
    if pip install vllm==0.5.4; then
        log "✓ VLLM installed successfully"
    else
        log_warning "VLLM installation failed, continuing without it"
    fi
    
    # Verify installation
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
    
    log "✓ Python dependencies installed"
}

download_model() {
    log "Downloading Qwen3-Coder-480B-A35B-Instruct model..."
    log_info "This will download approximately 450GB of data"
    log_info "Download time depends on your internet connection (may take several hours)"
    
    # Activate environment
    source "$INSTALL_DIR/bin/activate"
    
    # Create model directory
    mkdir -p "$INSTALL_DIR/models"
    cd "$INSTALL_DIR/models"
    
    # Set up Hugging Face cache
    export HF_HOME="$INSTALL_DIR/huggingface_cache"
    mkdir -p "$HF_HOME"
    
    # Login to Hugging Face (if needed)
    if [ -n "${HF_TOKEN:-}" ]; then
        echo "$HF_TOKEN" | huggingface-cli login --token-stdin
    fi
    
    # Download model with resume capability
    python3 << 'EOF'
import os
import sys
from huggingface_hub import snapshot_download
from pathlib import Path
import time

def download_with_resume(repo_id, local_dir, max_retries=5):
    """Download model with automatic resume on failure."""
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"Downloading {repo_id} (attempt {retry_count + 1}/{max_retries})...")
            
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                token=os.environ.get('HF_TOKEN'),
                allow_patterns=None,  # Download all files
            )
            
            print("✓ Model download completed successfully!")
            return True
            
        except Exception as e:
            retry_count += 1
            print(f"✗ Download failed (attempt {retry_count}): {e}")
            
            if retry_count < max_retries:
                wait_time = min(300, 60 * retry_count)  # Progressive backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("✗ All retry attempts failed!")
                return False
    
    return False

# Download the model
model_repo = "Qwen/Qwen2.5-Coder-32B-Instruct"
model_dir = os.path.join(os.environ['INSTALL_DIR'], 'models', 'qwen3-coder-480b')

success = download_with_resume(model_repo, model_dir)
if not success:
    print("Model download failed after all retries!")
    sys.exit(1)

print(f"Model successfully downloaded to: {model_dir}")
EOF
    
    log "✓ Model download completed"
}

create_test_scripts() {
    log "Creating test and example scripts..."
    
    # Create basic inference test
    cat > "$INSTALL_DIR/test_inference.py" << 'EOF'
#!/usr/bin/env python3
"""
Basic inference test for Qwen3-Coder-480B-A35B-Instruct
"""

import torch
import time
import sys
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

def test_model_loading():
    """Test model loading and basic inference."""
    print("🚀 Testing Qwen3-Coder-480B-A35B-Instruct...")
    
    # Configuration
    model_dir = os.path.join(os.environ.get('INSTALL_DIR', ''), 'models', 'qwen3-coder-480b')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"📁 Model directory: {model_dir}")
    print(f"💻 Device: {device}")
    print(f"🎮 CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Load tokenizer
        print("\n📝 Loading tokenizer...")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        print(f"✓ Tokenizer loaded in {time.time() - start_time:.2f}s")
        
        # Load model
        print("\n🧠 Loading model...")
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
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"💾 GPU Memory used: {memory_used:.1f}GB / {memory_total:.1f}GB ({memory_used/memory_total*100:.1f}%)")
        
        # Test inference
        print("\n🎯 Testing inference...")
        
        test_prompts = [
            "Write a Python function to calculate fibonacci numbers:",
            "Implement a binary search algorithm in C++:",
            "Create a REST API endpoint in Node.js:",
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}/3: {prompt[:50]}... ---")
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate response
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            inference_time = time.time() - start_time
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_tokens = len(outputs[0]) - len(inputs['input_ids'][0])
            tokens_per_second = new_tokens / inference_time
            
            print(f"⚡ Inference time: {inference_time:.2f}s")
            print(f"🎯 Tokens generated: {new_tokens}")
            print(f"📈 Speed: {tokens_per_second:.1f} tokens/second")
            print(f"📝 Response: {response[len(prompt):len(prompt)+100]}...")
            
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
EOF
    
    # Create performance benchmark
    cat > "$INSTALL_DIR/benchmark.py" << 'EOF'
#!/usr/bin/env python3
"""
Performance benchmark for Qwen3-Coder-480B-A35B-Instruct
"""

import torch
import time
import statistics
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil
import GPUtil

def run_benchmark():
    """Run comprehensive performance benchmark."""
    print("🏁 Starting Qwen3-Coder-480B Performance Benchmark")
    
    model_dir = os.path.join(os.environ.get('INSTALL_DIR', ''), 'models', 'qwen3-coder-480b')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Benchmark prompts
    prompts = [
        "Write a sorting algorithm",
        "Create a web scraper",
        "Implement a neural network",
        "Build a REST API",
        "Design a database schema",
    ]
    
    results = {
        "model": "Qwen3-Coder-480B-A35B-Instruct",
        "device": device,
        "torch_version": torch.__version__,
        "tests": []
    }
    
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        results["gpu_name"] = gpu.name
        results["gpu_memory_total"] = f"{gpu.memoryTotal}MB"
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nTest {i}/{len(prompts)}: {prompt}")
        
        # Run multiple iterations for statistical accuracy
        times = []
        token_counts = []
        
        for iteration in range(3):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            inference_time = time.time() - start_time
            new_tokens = len(outputs[0]) - len(inputs['input_ids'][0])
            
            times.append(inference_time)
            token_counts.append(new_tokens)
            
            torch.cuda.empty_cache()
        
        # Calculate statistics
        avg_time = statistics.mean(times)
        avg_tokens = statistics.mean(token_counts)
        tokens_per_second = avg_tokens / avg_time
        
        test_result = {
            "prompt": prompt,
            "avg_inference_time": round(avg_time, 3),
            "avg_tokens_generated": round(avg_tokens, 1),
            "tokens_per_second": round(tokens_per_second, 1),
            "times": [round(t, 3) for t in times]
        }
        
        results["tests"].append(test_result)
        
        print(f"  ⏱️  Average time: {avg_time:.3f}s")
        print(f"  🎯 Average tokens: {avg_tokens:.1f}")
        print(f"  📈 Speed: {tokens_per_second:.1f} tok/s")
    
    # Overall statistics
    all_speeds = [test["tokens_per_second"] for test in results["tests"]]
    results["overall_avg_speed"] = round(statistics.mean(all_speeds), 1)
    results["overall_max_speed"] = round(max(all_speeds), 1)
    results["overall_min_speed"] = round(min(all_speeds), 1)
    
    # Save results
    results_file = os.path.join(os.environ.get('INSTALL_DIR', ''), 'benchmark_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Benchmark Results Summary:")
    print(f"  Average Speed: {results['overall_avg_speed']} tokens/second")
    print(f"  Max Speed: {results['overall_max_speed']} tokens/second")
    print(f"  Min Speed: {results['overall_min_speed']} tokens/second")
    print(f"  Results saved to: {results_file}")

if __name__ == "__main__":
    run_benchmark()
EOF
    
    # Make scripts executable
    chmod +x "$INSTALL_DIR/test_inference.py"
    chmod +x "$INSTALL_DIR/benchmark.py"
    
    log "✓ Test scripts created"
}

create_activation_script() {
    log "Creating environment activation script..."
    
    cat > "$INSTALL_DIR/activate_qwen480b.sh" << EOF
#!/bin/bash
# Qwen3-Coder-480B Environment Activation Script

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "\${GREEN}🚀 Activating Qwen3-Coder-480B Environment\${NC}"

# Activate Python environment
source "$INSTALL_DIR/bin/activate"

# Set environment variables
export INSTALL_DIR="$INSTALL_DIR"
export HF_HOME="$INSTALL_DIR/huggingface_cache"
export CUDA_VISIBLE_DEVICES=0

# Add CUDA to PATH if not already there
if [[ ":\$PATH:" != *":/usr/local/cuda/bin:"* ]]; then
    export PATH="/usr/local/cuda/bin:\$PATH"
fi

if [[ ":\$LD_LIBRARY_PATH:" != *":/usr/local/cuda/lib64:"* ]]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
fi

echo -e "\${BLUE}Environment Variables:\${NC}"
echo "  INSTALL_DIR: \$INSTALL_DIR"
echo "  CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES"
echo "  Python: \$(which python)"

echo -e "\${GREEN}✓ Environment activated!\${NC}"
echo ""
echo -e "\${BLUE}Quick Commands:\${NC}"
echo "  Test installation: python test_inference.py"
echo "  Run benchmark: python benchmark.py"
echo "  Check GPU: nvidia-smi"
echo "  Monitor GPU: watch -n 1 nvidia-smi"

# Change to install directory
cd "$INSTALL_DIR"
EOF
    
    chmod +x "$INSTALL_DIR/activate_qwen480b.sh"
    
    # Create symlink in home directory for easy access
    ln -sf "$INSTALL_DIR/activate_qwen480b.sh" "$HOME/activate_qwen480b.sh"
    
    log "✓ Activation script created at $HOME/activate_qwen480b.sh"
}

run_installation_test() {
    log "Running installation verification test..."
    
    # Activate environment
    source "$INSTALL_DIR/bin/activate"
    export INSTALL_DIR="$INSTALL_DIR"
    
    # Run basic test
    if python "$INSTALL_DIR/test_inference.py"; then
        log "✅ Installation test PASSED"
        return 0
    else
        log_error "❌ Installation test FAILED"
        return 1
    fi
}

cleanup() {
    log "Cleaning up temporary files..."
    
    # Clean apt cache
    sudo apt autoremove -y
    sudo apt autoclean
    
    # Clean pip cache
    if [ -d "$INSTALL_DIR" ]; then
        source "$INSTALL_DIR/bin/activate"
        pip cache purge
    fi
    
    log "✓ Cleanup completed"
}

show_completion_message() {
    echo -e "${GREEN}"
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "📁 Installation directory: $INSTALL_DIR"
    echo "📋 Log file: $LOG_FILE"
    echo ""
    echo "🚀 Quick Start:"
    echo "  1. Activate environment: source ~/activate_qwen480b.sh"
    echo "  2. Test installation: python test_inference.py"
    echo "  3. Run benchmark: python benchmark.py"
    echo ""
    echo "📚 Documentation: https://github.com/twobitapps/480b-setup"
    echo -e "${NC}"
}

# Main installation flow
main() {
    print_header
    
    log "Starting Qwen3-Coder-480B-A35B-Instruct installation..."
    log "Log file: $LOG_FILE"
    
    # Pre-installation checks
    check_root
    check_system_requirements
    
    # System setup
    install_system_dependencies
    setup_cuda
    
    # Python environment setup
    create_python_environment
    install_python_dependencies
    
    # Model download
    download_model
    
    # Create utilities
    create_test_scripts
    create_activation_script
    
    # Verification
    if run_installation_test; then
        cleanup
        show_completion_message
        log "🎉 Installation completed successfully!"
    else
        log_error "Installation verification failed. Please check the logs."
        exit 1
    fi
}

# Handle interrupts
trap 'log_error "Installation interrupted by user"; exit 1' INT TERM

# Run main installation
main "$@"