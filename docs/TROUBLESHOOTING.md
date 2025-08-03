# Troubleshooting Guide

Common issues and solutions for Qwen3-Coder-480B-A35B-Instruct installation and usage.

## 🔧 Installation Issues

### GPU Memory Errors

**Error**: `CUDA out of memory`
```
RuntimeError: CUDA out of memory. Tried to allocate X GB (GPU 0; Y GB total capacity)
```

**Solutions**:
1. **Check GPU memory usage**: `nvidia-smi`
2. **Kill existing processes**: `sudo pkill -f python`
3. **Restart GPU**: `sudo nvidia-smi --gpu-reset`
4. **Reduce batch size** or use model sharding
5. **Enable CPU offloading**:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       device_map="auto",
       offload_folder="./offload",
       offload_state_dict=True
   )
   ```

### NumPy Version Conflicts

**Error**: `numpy.dtype size changed, may indicate binary incompatibility`

**Solutions**:
1. **Clean installation**:
   ```bash
   rm -rf ~/qwen480b_env
   pip cache purge
   ./install.sh
   ```

2. **Manual NumPy fix**:
   ```bash
   source ~/qwen480b_env/bin/activate
   pip uninstall numpy -y
   pip install numpy==1.24.4
   ```

### CUDA Installation Issues

**Error**: `NVCC not found` or `CUDA version mismatch`

**Solutions**:
1. **Install CUDA 12.1**:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt update
   sudo apt install cuda-toolkit-12-1
   ```

2. **Update PATH**:
   ```bash
   echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Verify installation**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

### Model Download Failures

**Error**: `Connection timeout` or `HTTP 403/404 errors`

**Solutions**:
1. **Check internet connection**: `ping huggingface.co`
2. **Use Hugging Face token** (if model requires authentication):
   ```bash
   export HF_TOKEN="your_token_here"
   ```
3. **Manual download with resume**:
   ```bash
   git clone https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct
   ```
4. **Use mirror servers**:
   ```bash
   export HF_ENDPOINT="https://hf-mirror.com"
   ```

## 🐛 Runtime Issues

### Model Loading Failures

**Error**: `Can't load model` or `Config file not found`

**Solutions**:
1. **Verify model directory**:
   ```bash
   ls -la ~/qwen480b_env/models/qwen3-coder-480b/
   ```

2. **Check file permissions**:
   ```bash
   chmod -R 755 ~/qwen480b_env/models/
   ```

3. **Re-download corrupted files**:
   ```bash
   cd ~/qwen480b_env/models/qwen3-coder-480b/
   git lfs pull
   ```

### Slow Inference

**Issue**: Model takes too long to generate responses

**Solutions**:
1. **Enable optimizations**:
   ```python
   # Use torch.compile (PyTorch 2.0+)
   model = torch.compile(model, mode="reduce-overhead")
   
   # Use better data types
   model = model.half()  # FP16
   ```

2. **Adjust generation parameters**:
   ```python
   outputs = model.generate(
       **inputs,
       max_new_tokens=100,  # Reduce if too slow
       do_sample=False,     # Use greedy decoding
       num_beams=1,         # Disable beam search
   )
   ```

3. **Use VLLM** (if installed):
   ```python
   from vllm import LLM, SamplingParams
   
   llm = LLM(model="path/to/model")
   outputs = llm.generate(prompts, SamplingParams(temperature=0.7))
   ```

### Memory Leaks

**Issue**: Memory usage keeps increasing

**Solutions**:
1. **Clear GPU cache regularly**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

2. **Use context managers**:
   ```python
   with torch.no_grad():
       outputs = model.generate(**inputs)
   ```

3. **Delete unused variables**:
   ```python
   del outputs, inputs
   import gc
   gc.collect()
   ```

## 🔍 Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# For transformers
import transformers
transformers.logging.set_verbosity_debug()
```

### Check System Resources

```bash
# GPU usage
nvidia-smi -l 1

# Memory usage
htop

# Disk usage
df -h

# Process monitoring
ps aux | grep python
```

### Model Information

```python
# Check model config
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_path)
print(config)

# Check model size
import torch
model_size = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {model_size:,}")
```

## 📊 Performance Tuning

### Optimize for Speed

1. **Use FP16 precision**:
   ```python
   model = model.half()
   ```

2. **Enable torch.compile**:
   ```python
   model = torch.compile(model)
   ```

3. **Optimize tokenizer**:
   ```python
   tokenizer.pad_token = tokenizer.eos_token
   tokenizer.padding_side = "left"
   ```

### Optimize for Memory

1. **Use model sharding**:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       device_map="auto",
       torch_dtype=torch.float16,
       low_cpu_mem_usage=True,
   )
   ```

2. **Enable gradient checkpointing**:
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **Use CPU offloading**:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       device_map="auto",
       offload_folder="./offload"
   )
   ```

## 🆘 Getting Help

### Collect Debug Information

Before reporting issues, collect this information:

```bash
# System info
./scripts/system_check.sh > debug_info.txt

# Python environment
pip list >> debug_info.txt

# GPU info
nvidia-smi >> debug_info.txt

# Error logs
tail -100 ~/qwen480b_install.log >> debug_info.txt
```

### Report Issues

1. **Check existing issues**: [GitHub Issues](https://github.com/twobitapps/480b-setup/issues)
2. **Create new issue** with:
   - Clear description of the problem
   - Steps to reproduce
   - Error messages and logs
   - System information from debug_info.txt

### Community Support

- **Discussions**: [GitHub Discussions](https://github.com/twobitapps/480b-setup/discussions)
- **Documentation**: [Project Wiki](https://github.com/twobitapps/480b-setup/wiki)

## 🔧 Advanced Debugging

### Memory Profiling

```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    outputs = model.generate(**inputs)

print(prof.key_averages().table(sort_by="cuda_memory_usage"))
```

### Performance Profiling

```python
import time
import torch

# Warm up
for _ in range(3):
    _ = model.generate(**inputs, max_new_tokens=10)

# Benchmark
times = []
for _ in range(10):
    start = time.time()
    outputs = model.generate(**inputs, max_new_tokens=50)
    times.append(time.time() - start)

print(f"Average time: {sum(times)/len(times):.2f}s")
```

### Network Debugging

```bash
# Test Hugging Face connectivity
curl -v https://huggingface.co

# Test download speed
wget --progress=bar --timeout=30 https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct/resolve/main/README.md

# Check DNS resolution
nslookup huggingface.co
```