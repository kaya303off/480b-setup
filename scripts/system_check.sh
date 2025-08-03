#!/bin/bash

# System Requirements Verification Script
# For Qwen3-Coder-480B-A35B-Instruct Installation

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Requirements
MIN_UBUNTU_VERSION="20.04"
MIN_DISK_SPACE_GB=500
MIN_MEMORY_GB=32
RECOMMENDED_MEMORY_GB=64
MIN_GPU_MEMORY_MB=70000
RECOMMENDED_GPU_MEMORY_MB=80000

print_header() {
    echo -e "${BLUE}🔍 Qwen3-Coder-480B System Requirements Check${NC}"
    echo "=============================================="
}

check_pass() {
    echo -e "  ${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "  ${RED}✗${NC} $1"
}

check_warning() {
    echo -e "  ${YELLOW}⚠${NC} $1"
}

check_info() {
    echo -e "  ${BLUE}ℹ${NC} $1"
}

check_operating_system() {
    echo -e "\n${BLUE}Operating System:${NC}"
    
    # Check if running on Linux
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        check_fail "Not running on Linux"
        return 1
    fi
    
    # Check if Ubuntu
    if ! grep -q "Ubuntu" /etc/os-release 2>/dev/null; then
        check_fail "Not running on Ubuntu"
        check_info "This script is optimized for Ubuntu 20.04+"
        return 1
    fi
    
    # Check Ubuntu version
    local ubuntu_version
    ubuntu_version=$(lsb_release -rs 2>/dev/null || echo "unknown")
    
    if [[ "$ubuntu_version" == "unknown" ]]; then
        check_warning "Cannot determine Ubuntu version"
        return 1
    fi
    
    # Compare versions
    if (( $(echo "$ubuntu_version >= $MIN_UBUNTU_VERSION" | bc -l) )); then
        check_pass "Ubuntu $ubuntu_version (✓ >= $MIN_UBUNTU_VERSION)"
    else
        check_fail "Ubuntu $ubuntu_version (✗ < $MIN_UBUNTU_VERSION required)"
        return 1
    fi
    
    # Check architecture
    local arch
    arch=$(uname -m)
    if [[ "$arch" == "x86_64" ]]; then
        check_pass "Architecture: $arch"
    else
        check_warning "Architecture: $arch (x86_64 recommended)"
    fi
    
    return 0
}

check_cpu() {
    echo -e "\n${BLUE}CPU:${NC}"
    
    local cpu_cores
    cpu_cores=$(nproc)
    
    if [[ $cpu_cores -ge 16 ]]; then
        check_pass "CPU cores: $cpu_cores (✓ >= 16 recommended)"
    elif [[ $cpu_cores -ge 8 ]]; then
        check_warning "CPU cores: $cpu_cores (8-15, 16+ recommended)"
    else
        check_warning "CPU cores: $cpu_cores (< 8, may be slow)"
    fi
    
    # CPU model
    local cpu_model
    cpu_model=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
    check_info "CPU model: $cpu_model"
    
    return 0
}

check_memory() {
    echo -e "\n${BLUE}Memory:${NC}"
    
    local total_mem_kb
    total_mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local total_mem_gb
    total_mem_gb=$((total_mem_kb / 1024 / 1024))
    
    if [[ $total_mem_gb -ge $RECOMMENDED_MEMORY_GB ]]; then
        check_pass "System RAM: ${total_mem_gb}GB (✓ >= ${RECOMMENDED_MEMORY_GB}GB recommended)"
    elif [[ $total_mem_gb -ge $MIN_MEMORY_GB ]]; then
        check_warning "System RAM: ${total_mem_gb}GB (${MIN_MEMORY_GB}-${RECOMMENDED_MEMORY_GB}GB, ${RECOMMENDED_MEMORY_GB}GB+ recommended)"
    else
        check_fail "System RAM: ${total_mem_gb}GB (✗ < ${MIN_MEMORY_GB}GB minimum)"
        return 1
    fi
    
    # Available memory
    local available_mem_kb
    available_mem_kb=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    local available_mem_gb
    available_mem_gb=$((available_mem_kb / 1024 / 1024))
    
    check_info "Available RAM: ${available_mem_gb}GB"
    
    # Swap
    local swap_total_kb
    swap_total_kb=$(grep SwapTotal /proc/meminfo | awk '{print $2}')
    local swap_total_gb
    swap_total_gb=$((swap_total_kb / 1024 / 1024))
    
    if [[ $swap_total_gb -gt 0 ]]; then
        check_info "Swap: ${swap_total_gb}GB"
    else
        check_warning "No swap configured (may cause OOM issues)"
    fi
    
    return 0
}

check_disk_space() {
    echo -e "\n${BLUE}Disk Space:${NC}"
    
    local home_disk_info
    home_disk_info=$(df -BG "$HOME" | awk 'NR==2 {print $2, $3, $4}')
    read -r total_space used_space available_space <<< "$home_disk_info"
    
    # Remove 'G' suffix and convert to numbers
    total_gb=${total_space%G}
    used_gb=${used_space%G}
    available_gb=${available_space%G}
    
    if [[ $available_gb -ge $MIN_DISK_SPACE_GB ]]; then
        check_pass "Available space: ${available_gb}GB (✓ >= ${MIN_DISK_SPACE_GB}GB required)"
    else
        check_fail "Available space: ${available_gb}GB (✗ < ${MIN_DISK_SPACE_GB}GB required)"
        return 1
    fi
    
    check_info "Total space: ${total_gb}GB"
    check_info "Used space: ${used_gb}GB"
    
    # Check if SSD (try to detect)
    local disk_device
    disk_device=$(df "$HOME" | awk 'NR==2 {print $1}' | sed 's/[0-9]*$//')
    
    if [[ -f "/sys/block/$(basename "$disk_device")/queue/rotational" ]]; then
        local is_rotational
        is_rotational=$(cat "/sys/block/$(basename "$disk_device")/queue/rotational")
        if [[ "$is_rotational" == "0" ]]; then
            check_pass "Storage type: SSD"
        else
            check_warning "Storage type: HDD (SSD recommended for better performance)"
        fi
    fi
    
    return 0
}

check_gpu() {
    echo -e "\n${BLUE}GPU:${NC}"
    
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        check_fail "NVIDIA GPU driver not found"
        check_info "Please install NVIDIA drivers first"
        return 1
    fi
    
    # Check GPU details
    local gpu_info
    if ! gpu_info=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>/dev/null); then
        check_fail "Cannot query GPU information"
        return 1
    fi
    
    local gpu_count=0
    while IFS=',' read -r gpu_name gpu_memory driver_version; do
        gpu_count=$((gpu_count + 1))
        
        # Trim whitespace
        gpu_name=$(echo "$gpu_name" | xargs)
        gpu_memory=$(echo "$gpu_memory" | xargs)
        driver_version=$(echo "$driver_version" | xargs)
        
        check_info "GPU $gpu_count: $gpu_name"
        check_info "Driver version: $driver_version"
        
        if [[ $gpu_memory -ge $RECOMMENDED_GPU_MEMORY_MB ]]; then
            check_pass "GPU Memory: ${gpu_memory}MB (✓ >= ${RECOMMENDED_GPU_MEMORY_MB}MB recommended)"
        elif [[ $gpu_memory -ge $MIN_GPU_MEMORY_MB ]]; then
            check_warning "GPU Memory: ${gpu_memory}MB (${MIN_GPU_MEMORY_MB}-${RECOMMENDED_GPU_MEMORY_MB}MB, may work but tight)"
        else
            check_fail "GPU Memory: ${gpu_memory}MB (✗ < ${MIN_GPU_MEMORY_MB}MB minimum)"
            return 1
        fi
        
        # Check for H100 or A100
        if [[ "$gpu_name" == *"H100"* ]]; then
            check_pass "GPU Type: H100 (optimal)"
        elif [[ "$gpu_name" == *"A100"* ]]; then
            check_pass "GPU Type: A100 (excellent)"
        elif [[ "$gpu_name" == *"RTX"* ]] && [[ $gpu_memory -ge $MIN_GPU_MEMORY_MB ]]; then
            check_warning "GPU Type: RTX (may work with sufficient memory)"
        else
            check_warning "GPU Type: $gpu_name (H100 or A100 recommended)"
        fi
        
    done <<< "$gpu_info"
    
    if [[ $gpu_count -eq 0 ]]; then
        check_fail "No NVIDIA GPUs detected"
        return 1
    fi
    
    # Check CUDA
    if command -v nvcc &> /dev/null; then
        local cuda_version
        cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        if (( $(echo "$cuda_version >= 11.8" | bc -l) )); then
            check_pass "CUDA version: $cuda_version (✓ >= 11.8)"
        else
            check_warning "CUDA version: $cuda_version (11.8+ recommended)"
        fi
    else
        check_warning "CUDA toolkit not found (will be installed)"
    fi
    
    return 0
}

check_network() {
    echo -e "\n${BLUE}Network:${NC}"
    
    # Check internet connectivity
    if ping -c 1 google.com &> /dev/null; then
        check_pass "Internet connectivity"
    else
        check_fail "No internet connectivity"
        return 1
    fi
    
    # Check Hugging Face connectivity
    if curl -s --connect-timeout 5 https://huggingface.co &> /dev/null; then
        check_pass "Hugging Face reachable"
    else
        check_warning "Cannot reach Hugging Face (may need proxy config)"
    fi
    
    return 0
}

check_python() {
    echo -e "\n${BLUE}Python:${NC}"
    
    # Check Python 3
    if command -v python3 &> /dev/null; then
        local python_version
        python_version=$(python3 --version | awk '{print $2}')
        check_pass "Python 3: $python_version"
        
        # Check if version is compatible
        local major minor
        IFS='.' read -r major minor _ <<< "$python_version"
        if [[ $major -eq 3 ]] && [[ $minor -ge 8 ]] && [[ $minor -le 11 ]]; then
            check_pass "Python version compatible (3.8-3.11)"
        else
            check_warning "Python version $python_version (3.8-3.11 recommended)"
        fi
    else
        check_fail "Python 3 not found"
        return 1
    fi
    
    # Check pip
    if command -v pip3 &> /dev/null; then
        check_pass "pip3 available"
    else
        check_warning "pip3 not found (will be installed)"
    fi
    
    # Check venv
    if python3 -c "import venv" &> /dev/null; then
        check_pass "venv module available"
    else
        check_warning "venv module not found (will be installed)"
    fi
    
    return 0
}

check_dependencies() {
    echo -e "\n${BLUE}System Dependencies:${NC}"
    
    local required_packages=(
        "git"
        "curl"
        "wget"
        "build-essential"
        "cmake"
    )
    
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if command -v "$package" &> /dev/null || dpkg -l "$package" &> /dev/null; then
            check_pass "$package installed"
        else
            check_warning "$package not found (will be installed)"
            missing_packages+=("$package")
        fi
    done
    
    # Check Git LFS
    if command -v git-lfs &> /dev/null; then
        check_pass "Git LFS installed"
    else
        check_warning "Git LFS not found (will be installed)"
    fi
    
    return 0
}

print_summary() {
    echo -e "\n${BLUE}Summary:${NC}"
    echo "========"
    
    local total_checks=8
    local passed_checks=0
    
    # Count passed checks (simplified)
    if check_operating_system &>/dev/null; then ((passed_checks++)); fi
    if check_cpu &>/dev/null; then ((passed_checks++)); fi
    if check_memory &>/dev/null; then ((passed_checks++)); fi
    if check_disk_space &>/dev/null; then ((passed_checks++)); fi
    if check_gpu &>/dev/null; then ((passed_checks++)); fi
    if check_network &>/dev/null; then ((passed_checks++)); fi
    if check_python &>/dev/null; then ((passed_checks++)); fi
    if check_dependencies &>/dev/null; then ((passed_checks++)); fi
    
    if [[ $passed_checks -eq $total_checks ]]; then
        echo -e "${GREEN}✅ System ready for Qwen3-Coder-480B installation!${NC}"
        return 0
    elif [[ $passed_checks -ge $((total_checks - 2)) ]]; then
        echo -e "${YELLOW}⚠️  System mostly ready - check warnings above${NC}"
        return 0
    else
        echo -e "${RED}❌ System not ready - address failures above${NC}"
        return 1
    fi
}

main() {
    print_header
    
    local overall_status=0
    
    check_operating_system || overall_status=1
    check_cpu || overall_status=1
    check_memory || overall_status=1
    check_disk_space || overall_status=1
    check_gpu || overall_status=1
    check_network || overall_status=1
    check_python || overall_status=1
    check_dependencies || overall_status=1
    
    print_summary
    
    if [[ $overall_status -eq 0 ]]; then
        echo -e "\n${GREEN}🚀 Ready to install! Run: ./install.sh${NC}"
    else
        echo -e "\n${YELLOW}💡 Fix the issues above, then run this check again${NC}"
    fi
    
    return $overall_status
}

# Run main function
main "$@"