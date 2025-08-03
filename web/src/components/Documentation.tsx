export default function Documentation() {
  const docs = [
    {
      title: "README.md",
      description: "Main documentation with quick start guide and overview",
      icon: "📄",
      url: "https://github.com/twobitapps/480b-setup/blob/main/README.md",
      type: "guide"
    },
    {
      title: "MANUAL_INSTALL.md",
      description: "Detailed step-by-step manual installation instructions",
      icon: "📖",
      url: "https://github.com/twobitapps/480b-setup/blob/main/MANUAL_INSTALL.md",
      type: "guide"
    },
    {
      title: "TROUBLESHOOTING.md",
      description: "Common issues, solutions, and debugging techniques",
      icon: "🛠️",
      url: "https://github.com/twobitapps/480b-setup/blob/main/docs/TROUBLESHOOTING.md",
      type: "reference"
    },
    {
      title: "install.sh",
      description: "Main automated installation script with error handling",
      icon: "🤖",
      url: "https://github.com/twobitapps/480b-setup/blob/main/install.sh",
      type: "script"
    },
    {
      title: "system_check.sh",
      description: "System requirements verification script",
      icon: "🔍",
      url: "https://github.com/twobitapps/480b-setup/blob/main/scripts/system_check.sh",
      type: "script"
    },
    {
      title: "test_installation.sh",
      description: "Installation validation and testing script",
      icon: "🧪",
      url: "https://github.com/twobitapps/480b-setup/blob/main/scripts/test_installation.sh",
      type: "script"
    },
    {
      title: "basic_inference.py",
      description: "Example Python script for model inference",
      icon: "🐍",
      url: "https://github.com/twobitapps/480b-setup/blob/main/examples/basic_inference.py",
      type: "example"
    },
    {
      title: "requirements.txt",
      description: "Python dependencies with tested versions",
      icon: "📦",
      url: "https://github.com/twobitapps/480b-setup/blob/main/config/requirements.txt",
      type: "config"
    }
  ]

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'guide': return 'bg-blue-900/30 border-blue-600/30 text-blue-400'
      case 'reference': return 'bg-yellow-900/30 border-yellow-600/30 text-yellow-400'
      case 'script': return 'bg-green-900/30 border-green-600/30 text-green-400'
      case 'example': return 'bg-purple-900/30 border-purple-600/30 text-purple-400'
      case 'config': return 'bg-gray-900/30 border-gray-600/30 text-gray-400'
      default: return 'bg-gray-900/30 border-gray-600/30 text-gray-400'
    }
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-terminal-secondary mb-2">
          📚 Documentation
        </h2>
        <p className="text-terminal-primary">
          Comprehensive guides, scripts, and examples for Qwen3-Coder-480B setup
        </p>
      </div>

      {/* Quick Links */}
      <div className="grid md:grid-cols-3 gap-4">
        <a 
          href="https://github.com/twobitapps/480b-setup"
          target="_blank"
          rel="noopener noreferrer"
          className="bg-gray-900 border border-gray-700 rounded-lg p-4 hover:border-terminal-primary transition-colors"
        >
          <div className="text-center">
            <div className="text-2xl mb-2">🚀</div>
            <h3 className="font-semibold text-terminal-secondary">GitHub Repository</h3>
            <p className="text-sm text-gray-400 mt-1">Complete source code and documentation</p>
          </div>
        </a>
        
        <a 
          href="https://github.com/twobitapps/480b-setup/releases"
          target="_blank"
          rel="noopener noreferrer"
          className="bg-gray-900 border border-gray-700 rounded-lg p-4 hover:border-terminal-primary transition-colors"
        >
          <div className="text-center">
            <div className="text-2xl mb-2">📦</div>
            <h3 className="font-semibold text-terminal-secondary">Releases</h3>
            <p className="text-sm text-gray-400 mt-1">Download specific versions</p>
          </div>
        </a>
        
        <a 
          href="https://github.com/twobitapps/480b-setup/issues"
          target="_blank"
          rel="noopener noreferrer"
          className="bg-gray-900 border border-gray-700 rounded-lg p-4 hover:border-terminal-primary transition-colors"
        >
          <div className="text-center">
            <div className="text-2xl mb-2">💬</div>
            <h3 className="font-semibold text-terminal-secondary">Support</h3>
            <p className="text-sm text-gray-400 mt-1">Issues and discussions</p>
          </div>
        </a>
      </div>

      {/* Documentation Files */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-terminal-accent mb-4 flex items-center gap-2">
          📋 Documentation Files
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          {docs.map((doc, index) => (
            <a
              key={index}
              href={doc.url}
              target="_blank"
              rel="noopener noreferrer"
              className="block bg-gray-800 border border-gray-700 rounded-lg p-4 hover:border-terminal-primary transition-colors"
            >
              <div className="flex items-start gap-3">
                <div className="text-xl">{doc.icon}</div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-semibold text-terminal-primary">{doc.title}</h4>
                    <span className={`text-xs px-2 py-1 rounded border ${getTypeColor(doc.type)}`}>
                      {doc.type}
                    </span>
                  </div>
                  <p className="text-sm text-gray-400">{doc.description}</p>
                </div>
              </div>
            </a>
          ))}
        </div>
      </div>

      {/* API Reference */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-terminal-accent mb-4 flex items-center gap-2">
          🔧 Usage Examples
        </h3>
        <div className="space-y-4">
          <div>
            <h4 className="font-semibold text-terminal-secondary mb-2">Basic Model Usage</h4>
            <div className="code-block">
              <code className="text-terminal-primary text-sm">{`# Activate environment
source ~/activate_qwen480b.sh

# Load and use the model
python3 << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model
model_path = "~/qwen480b_env/models/qwen3-coder-480b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Generate code
prompt = "Write a Python function to sort a list:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
EOF`}</code>
            </div>
          </div>
          
          <div>
            <h4 className="font-semibold text-terminal-secondary mb-2">Performance Monitoring</h4>
            <div className="code-block">
              <code className="text-terminal-primary text-sm">{`# Monitor GPU usage
watch -n 1 nvidia-smi

# Run performance benchmark
python benchmark.py

# Check system resources
htop
gpustat -i 1`}</code>
            </div>
          </div>
        </div>
      </div>

      {/* Community Resources */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-terminal-accent mb-4 flex items-center gap-2">
          🌟 Community & Resources
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold text-terminal-secondary mb-3">Official Resources</h4>
            <div className="space-y-2 text-sm">
              <a 
                href="https://github.com/QwenLM/Qwen2.5-Coder"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors"
              >
                🔗 Official Qwen Repository
              </a>
              <a 
                href="https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors"
              >
                🤗 Hugging Face Model
              </a>
              <a 
                href="https://github.com/twobitapps/hyperdev-1"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors"
              >
                🎨 Model Comparison Demo
              </a>
            </div>
          </div>
          <div>
            <h4 className="font-semibold text-terminal-secondary mb-3">Support Channels</h4>
            <div className="space-y-2 text-sm">
              <a 
                href="https://github.com/twobitapps/480b-setup/discussions"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors"
              >
                💬 GitHub Discussions
              </a>
              <a 
                href="https://github.com/twobitapps/480b-setup/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors"
              >
                🐛 Bug Reports
              </a>
              <a 
                href="https://github.com/twobitapps/480b-setup/wiki"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors"
              >
                📖 Community Wiki
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}