#!/bin/bash

# Installation Verification Script for Qwen3-Coder-480B-A35B-Instruct
# This script verifies that everything is installed correctly

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
INSTALL_DIR="${INSTALL_DIR:-$HOME/qwen480b_env}"
TEST_LOG="$HOME/qwen480b_test.log"

print_header() {
    echo -e "${BLUE}🧪 Qwen3-Coder-480B Installation Test${NC}"
    echo "====================================="
}

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$TEST_LOG"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$TEST_LOG"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$TEST_LOG"
}

check_pass() {
    echo -e "  ${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "  ${RED}✗${NC} $1"
    return 1
}

check_warning() {
    echo -e "  ${YELLOW}⚠${NC} $1"
}

test_environment() {
    log "Testing Python environment..."
    
    # Check if environment exists
    if [[ ! -d "$INSTALL_DIR" ]]; then
        check_fail "Installation directory not found: $INSTALL_DIR"
        return 1
    fi
    check_pass "Installation directory exists"
    
    # Check if virtual environment is valid
    if [[ ! -f "$INSTALL_DIR/bin/activate" ]]; then
        check_fail "Python virtual environment not found"
        return 1
    fi
    check_pass "Python virtual environment found"
    
    # Activate environment
    source "$INSTALL_DIR/bin/activate"
    
    # Check Python version
    local python_version
    python_version=$(python --version 2>&1 | awk '{print $2}')
    check_pass "Python version: $python_version"
    
    # Check if this is the correct environment
    local python_path
    python_path=$(which python)
    if [[ "$python_path" == "$INSTALL_DIR"* ]]; then
        check_pass "Using correct Python environment"
    else
        check_warning "Python path: $python_path (expected in $INSTALL_DIR)"
    fi
    
    return 0
}

test_gpu() {
    log "Testing GPU availability..."
    
    # Check NVIDIA driver
    if ! command -v nvidia-smi &> /dev/null; then
        check_fail "nvidia-smi not found"
        return 1
    fi
    check_pass "NVIDIA driver available"
    
    # Check GPU status
    if nvidia-smi &> /dev/null; then
        local gpu_count
        gpu_count=$(nvidia-smi -L | wc -l)
        check_pass "GPUs detected: $gpu_count"
        
        # Show GPU memory
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | while read -r line; do
            check_pass "GPU: $line"
        done
    else
        check_fail "nvidia-smi command failed"
        return 1
    fi
    
    return 0
}

test_python_packages() {
    log "Testing Python packages..."
    
    source "$INSTALL_DIR/bin/activate"
    
    # Core packages to test
    local packages=(
        "torch"
        "transformers"
        "accelerate"
        "tokenizers"
        "huggingface_hub"
    )
    
    for package in "${packages[@]}"; do
        if python -c "import $package" &> /dev/null; then
            local version
            version=$(python -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
            check_pass "$package version $version"
        else
            check_fail "$package not importable"
            return 1
        fi
    done
    
    return 0
}

test_torch_cuda() {
    log "Testing PyTorch CUDA support..."
    
    source "$INSTALL_DIR/bin/activate"
    
    # Test PyTorch CUDA
    local cuda_available
    cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    
    if [[ "$cuda_available" == "True" ]]; then
        check_pass "PyTorch CUDA support enabled"
        
        # CUDA version
        local cuda_version
        cuda_version=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
        check_pass "CUDA version: $cuda_version"
        
        # GPU count
        local gpu_count
        gpu_count=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
        check_pass "PyTorch can see $gpu_count GPU(s)"
        
        # Test tensor creation
        if python -c "import torch; x = torch.tensor([1.0]).cuda(); print('GPU tensor test passed')" &> /dev/null; then
            check_pass "GPU tensor creation successful"
        else
            check_fail "GPU tensor creation failed"
            return 1
        fi
        
    else
        check_fail "PyTorch CUDA support not available"
        return 1
    fi
    
    return 0
}

test_model_files() {
    log "Testing model files..."
    
    local model_dir="$INSTALL_DIR/models/qwen3-coder-480b"
    
    if [[ ! -d "$model_dir" ]]; then
        check_fail "Model directory not found: $model_dir"
        return 1
    fi
    check_pass "Model directory exists"
    
    # Check essential files
    local essential_files=(
        "config.json"
        "tokenizer.json"
        "tokenizer_config.json"
    )
    
    for file in "${essential_files[@]}"; do
        if [[ -f "$model_dir/$file" ]]; then
            check_pass "$file found"
        else
            check_fail "$file missing"
            return 1
        fi
    done
    
    # Check for model weights (should have .safetensors or .bin files)
    local weight_files
    weight_files=$(find "$model_dir" -name "*.safetensors" -o -name "*.bin" | wc -l)
    
    if [[ $weight_files -gt 0 ]]; then
        check_pass "Model weight files found: $weight_files"
    else
        check_fail "No model weight files found"
        return 1
    fi
    
    # Check total model size
    local model_size
    model_size=$(du -sh "$model_dir" | cut -f1)
    check_pass "Model size: $model_size"
    
    return 0
}

test_model_loading() {
    log "Testing model loading..."
    
    source "$INSTALL_DIR/bin/activate"
    export INSTALL_DIR="$INSTALL_DIR"
    
    # Create a simple test script
    cat > "$INSTALL_DIR/test_load.py" << 'EOF'
import sys
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

try:
    model_dir = os.path.join(os.environ['INSTALL_DIR'], 'models', 'qwen3-coder-480b')
    
    print("Loading tokenizer...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer_time = time.time() - start_time
    print(f"✓ Tokenizer loaded in {tokenizer_time:.2f}s")
    
    print("Loading model...")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model_time = time.time() - start_time
    print(f"✓ Model loaded in {model_time:.2f}s")
    
    # Check GPU memory
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1e9
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB")
    
    print("✓ Model loading test passed")
    
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    sys.exit(1)
EOF
    
    # Run the test
    if python "$INSTALL_DIR/test_load.py"; then
        check_pass "Model loading successful"
        rm -f "$INSTALL_DIR/test_load.py"
        return 0
    else
        check_fail "Model loading failed"
        rm -f "$INSTALL_DIR/test_load.py"
        return 1
    fi
}

test_basic_inference() {
    log "Testing basic inference..."
    
    source "$INSTALL_DIR/bin/activate"
    export INSTALL_DIR="$INSTALL_DIR"
    
    # Create inference test
    cat > "$INSTALL_DIR/test_inference_quick.py" << 'EOF'
import sys
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

try:
    model_dir = os.path.join(os.environ['INSTALL_DIR'], 'models', 'qwen3-coder-480b')
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Test prompt
    prompt = "def fibonacci(n):"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    inference_time = time.time() - start_time
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = response[len(prompt):]
    
    new_tokens = len(outputs[0]) - len(inputs['input_ids'][0])
    tokens_per_second = new_tokens / inference_time
    
    print(f"✓ Inference completed in {inference_time:.2f}s")
    print(f"✓ Generated {new_tokens} tokens at {tokens_per_second:.1f} tokens/s")
    print(f"✓ Sample output: {generated[:100]}...")
    
except Exception as e:
    print(f"✗ Inference test failed: {e}")
    sys.exit(1)
EOF
    
    # Run the test with timeout
    if timeout 300 python "$INSTALL_DIR/test_inference_quick.py"; then
        check_pass "Basic inference successful"
        rm -f "$INSTALL_DIR/test_inference_quick.py"
        return 0
    else
        check_fail "Basic inference failed or timed out"
        rm -f "$INSTALL_DIR/test_inference_quick.py"
        return 1
    fi
}

test_utilities() {
    log "Testing utility scripts..."
    
    # Check activation script
    if [[ -f "$HOME/activate_qwen480b.sh" ]]; then
        check_pass "Activation script available"
    else
        check_warning "Activation script not found"
    fi
    
    # Check example scripts
    if [[ -f "$INSTALL_DIR/test_inference.py" ]]; then
        check_pass "Test inference script available"
    else
        check_warning "Test inference script not found"
    fi
    
    if [[ -f "$INSTALL_DIR/benchmark.py" ]]; then
        check_pass "Benchmark script available"
    else
        check_warning "Benchmark script not found"
    fi
    
    return 0
}

run_performance_check() {
    log "Running quick performance check..."
    
    source "$INSTALL_DIR/bin/activate"
    
    # Quick GPU memory check
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Status:"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader | while read -r line; do
            echo "  $line"
        done
    fi
    
    # Python memory usage
    local python_memory
    python_memory=$(python -c "import psutil; print(f'{psutil.virtual_memory().used / 1e9:.1f}GB')" 2>/dev/null || echo "unknown")
    echo "Python process memory: $python_memory"
    
    return 0
}

print_summary() {
    local total_tests=8
    local passed_tests=0
    
    echo -e "\n${BLUE}Test Summary:${NC}"
    echo "============"
    
    # Count passed tests (simplified check)
    if test_environment &>/dev/null; then ((passed_tests++)); fi
    if test_gpu &>/dev/null; then ((passed_tests++)); fi
    if test_python_packages &>/dev/null; then ((passed_tests++)); fi
    if test_torch_cuda &>/dev/null; then ((passed_tests++)); fi
    if test_model_files &>/dev/null; then ((passed_tests++)); fi
    if test_model_loading &>/dev/null; then ((passed_tests++)); fi
    if test_basic_inference &>/dev/null; then ((passed_tests++)); fi
    if test_utilities &>/dev/null; then ((passed_tests++)); fi
    
    echo "Tests passed: $passed_tests/$total_tests"
    
    if [[ $passed_tests -eq $total_tests ]]; then
        echo -e "${GREEN}🎉 All tests passed! Installation is working correctly.${NC}"
        echo -e "\n${BLUE}Quick start:${NC}"
        echo "  source ~/activate_qwen480b.sh"
        echo "  python examples/basic_inference.py"
        return 0
    elif [[ $passed_tests -ge $((total_tests - 2)) ]]; then
        echo -e "${YELLOW}⚠️  Most tests passed, but check warnings above${NC}"
        return 0
    else
        echo -e "${RED}❌ Multiple test failures - installation may be incomplete${NC}"
        echo -e "Check the troubleshooting guide: docs/TROUBLESHOOTING.md"
        return 1
    fi
}

main() {
    print_header
    log "Starting installation verification tests..."
    log "Test log: $TEST_LOG"
    
    local overall_status=0
    
    test_environment || overall_status=1
    test_gpu || overall_status=1
    test_python_packages || overall_status=1
    test_torch_cuda || overall_status=1
    test_model_files || overall_status=1
    test_model_loading || overall_status=1
    test_basic_inference || overall_status=1
    test_utilities || overall_status=1
    
    run_performance_check
    print_summary
    
    log "Test completed with status: $overall_status"
    return $overall_status
}

# Handle interrupts
trap 'log "Test interrupted"; exit 1' INT TERM

# Run tests
main "$@"