# Qwen3-Coder-480B-A35B-Instruct Setup Guide

Complete automated installation guide for Qwen3-Coder-480B-A35B-Instruct model on Ubuntu systems.

## 🚀 Quick Start

```bash
# One-line installation
curl -fsSL https://raw.githubusercontent.com/twobitapps/480b-setup/main/install.sh | bash
```

## 📋 System Requirements

### Minimum Hardware Requirements
- **GPU**: NVIDIA H100 80GB HBM3 (recommended) or A100 80GB
- **RAM**: 64GB+ system RAM
- **Storage**: 500GB+ free space (450GB for model + dependencies)
- **CPU**: 16+ cores recommended
- **Network**: High-speed internet for initial model download

### Software Requirements
- **OS**: Ubuntu 20.04+ or 22.04 LTS (recommended)
- **Python**: 3.8-3.11 (3.10 recommended)
- **CUDA**: 12.1+ (will be installed automatically)
- **Git**: Latest version
- **Git LFS**: For large file handling

## 🔧 Installation Methods

### Method 1: Automated Script (Recommended)
```bash
./install.sh
```

### Method 2: Manual Step-by-Step
Follow the detailed instructions in [MANUAL_INSTALL.md](./MANUAL_INSTALL.md)

### Method 3: Docker Setup
```bash
docker-compose up -d
```

## 📁 Repository Structure

```
480b-setup/
├── README.md                 # This file
├── install.sh               # Main installation script
├── MANUAL_INSTALL.md        # Step-by-step manual guide
├── scripts/
│   ├── system_check.sh      # System requirements verification
│   ├── dependencies.sh      # Install system dependencies
│   ├── python_env.sh        # Python environment setup
│   ├── cuda_setup.sh        # CUDA installation
│   ├── model_download.sh    # Model download with resume
│   ├── test_installation.sh # Installation verification
│   └── benchmark.sh         # Performance testing
├── config/
│   ├── requirements.txt     # Python dependencies
│   ├── environment.yml      # Conda environment
│   └── model_config.json    # Model configuration
├── examples/
│   ├── basic_inference.py   # Simple inference example
│   ├── benchmark_test.py    # Performance benchmark
│   └── comparison_demo.py   # 480B vs 7B comparison
├── docker/
│   ├── Dockerfile           # Docker container setup
│   └── docker-compose.yml   # Docker Compose configuration
└── docs/
    ├── TROUBLESHOOTING.md   # Common issues and solutions
    ├── PERFORMANCE.md       # Performance tuning guide
    └── API_REFERENCE.md     # API usage documentation
```

## ⚡ Quick Verification

After installation, verify everything works:

```bash
# Run system verification
./scripts/test_installation.sh

# Run basic inference test
python examples/basic_inference.py

# Run performance benchmark
./scripts/benchmark.sh
```

## 🐛 Troubleshooting

If you encounter issues:

1. Check [TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md)
2. Run `./scripts/system_check.sh` to verify requirements
3. Check logs in `~/qwen480b_env/logs/`
4. Open an issue with detailed error logs

## 📊 Performance Expectations

On NVIDIA H100 80GB:
- **Model Loading**: ~2-3 minutes
- **First Inference**: ~10-15 seconds (cold start)
- **Subsequent Inference**: ~3-6 seconds
- **Tokens per Second**: 100-200 (depends on prompt complexity)
- **Memory Usage**: ~45-50GB VRAM

## 🔗 Related Projects

- [Qwen Model Comparison Platform](https://github.com/twobitapps/hyperdev-1) - Visual comparison demo
- [Original Qwen Repository](https://github.com/QwenLM/Qwen2.5-Coder)
- [Hugging Face Model](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)

## 📝 License

This setup guide is provided under MIT License. The Qwen model follows its own licensing terms.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test your changes thoroughly
4. Submit a pull request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/twobitapps/480b-setup/issues)
- **Discussions**: [GitHub Discussions](https://github.com/twobitapps/480b-setup/discussions)
- **Documentation**: [Wiki](https://github.com/twobitapps/480b-setup/wiki)

---

**⚠️ Note**: This is a large model requiring significant computational resources. Ensure your system meets the minimum requirements before installation.