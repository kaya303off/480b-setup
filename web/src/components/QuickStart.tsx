export default function QuickStart() {
  const installCommand = 'curl -fsSL https://raw.githubusercontent.com/twobitapps/480b-setup/main/install.sh | bash'
  
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-terminal-secondary mb-2">
          🚀 One-Click Installation
        </h2>
        <p className="text-terminal-primary">
          Get Qwen3-Coder-480B running in minutes with our automated installer
        </p>
      </div>

      {/* Quick Install */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-terminal-accent mb-4 flex items-center gap-2">
          ⚡ Quick Install
        </h3>
        <div className="code-block">
          <div className="flex items-center justify-between mb-2">
            <span className="text-terminal-secondary">$ </span>
            <button 
              onClick={() => navigator.clipboard.writeText(installCommand)}
              className="text-xs bg-gray-700 px-2 py-1 rounded hover:bg-gray-600 transition-colors"
            >
              Copy
            </button>
          </div>
          <code className="text-terminal-primary break-all">
            {installCommand}
          </code>
        </div>
        <p className="text-sm text-gray-400 mt-2">
          ⚠️ Requires: Ubuntu 20.04+, NVIDIA H100/A100 GPU, 64GB+ RAM, 500GB+ storage
        </p>
      </div>

      {/* Installation Methods */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
          <h4 className="font-semibold text-terminal-secondary mb-2 flex items-center gap-2">
            🤖 Automated
          </h4>
          <p className="text-sm text-gray-300 mb-3">
            Full automated installation with error recovery and progress tracking
          </p>
          <div className="code-block text-xs">
            ./install.sh
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
          <h4 className="font-semibold text-terminal-secondary mb-2 flex items-center gap-2">
            📖 Manual
          </h4>
          <p className="text-sm text-gray-300 mb-3">
            Step-by-step guide for custom installations and troubleshooting
          </p>
          <div className="code-block text-xs">
            MANUAL_INSTALL.md
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
          <h4 className="font-semibold text-terminal-secondary mb-2 flex items-center gap-2">
            🐳 Docker
          </h4>
          <p className="text-sm text-gray-300 mb-3">
            Containerized deployment with isolated environment
          </p>
          <div className="code-block text-xs">
            docker-compose up
          </div>
        </div>
      </div>

      {/* GitHub Repository */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-terminal-accent mb-4 flex items-center gap-2">
          📂 GitHub Repository
        </h3>
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <p className="text-gray-300 mb-2">
              Complete source code, documentation, and installation scripts
            </p>
            <div className="code-block">
              <code className="text-terminal-primary">
                git clone https://github.com/twobitapps/480b-setup.git
              </code>
            </div>
          </div>
          <div className="flex flex-col gap-2">
            <a 
              href="https://github.com/twobitapps/480b-setup" 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 bg-terminal-primary text-black px-4 py-2 rounded hover:bg-terminal-secondary transition-colors font-semibold text-center"
            >
              🔗 View on GitHub
            </a>
            <a 
              href="https://github.com/twobitapps/480b-setup/releases" 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 bg-gray-700 text-terminal-primary px-4 py-2 rounded hover:bg-gray-600 transition-colors text-center"
            >
              📦 Releases
            </a>
          </div>
        </div>
      </div>

      {/* Performance Expectations */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-terminal-accent mb-4 flex items-center gap-2">
          📊 Performance Expectations
        </h3>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
          <div className="text-center p-3 bg-gray-800 rounded">
            <div className="text-terminal-secondary font-semibold">Model Loading</div>
            <div className="text-terminal-primary">2-3 minutes</div>
          </div>
          <div className="text-center p-3 bg-gray-800 rounded">
            <div className="text-terminal-secondary font-semibold">First Inference</div>
            <div className="text-terminal-primary">10-15 seconds</div>
          </div>
          <div className="text-center p-3 bg-gray-800 rounded">
            <div className="text-terminal-secondary font-semibold">Speed</div>
            <div className="text-terminal-primary">100-200 tok/s</div>
          </div>
          <div className="text-center p-3 bg-gray-800 rounded">
            <div className="text-terminal-secondary font-semibold">GPU Memory</div>
            <div className="text-terminal-primary">~45-50GB</div>
          </div>
        </div>
        <p className="text-xs text-gray-400 mt-3 text-center">
          * Performance on NVIDIA H100 80GB HBM3
        </p>
      </div>
    </div>
  )
}