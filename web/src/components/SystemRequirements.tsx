export default function SystemRequirements() {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-terminal-secondary mb-2">
          📋 System Requirements
        </h2>
        <p className="text-terminal-primary">
          Ensure your system meets these requirements before installation
        </p>
      </div>

      {/* Hardware Requirements */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-terminal-accent mb-4 flex items-center gap-2">
          🖥️ Hardware Requirements
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold text-terminal-secondary mb-3">Minimum Requirements</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-300">GPU:</span>
                <span className="text-terminal-primary">NVIDIA A100 80GB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">RAM:</span>
                <span className="text-terminal-primary">64GB system memory</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Storage:</span>
                <span className="text-terminal-primary">500GB free space</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">CPU:</span>
                <span className="text-terminal-primary">16+ cores</span>
              </div>
            </div>
          </div>
          <div>
            <h4 className="font-semibold text-terminal-secondary mb-3">Recommended Setup</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-300">GPU:</span>
                <span className="text-terminal-accent">NVIDIA H100 80GB HBM3</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">RAM:</span>
                <span className="text-terminal-accent">128GB+ system memory</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Storage:</span>
                <span className="text-terminal-accent">1TB+ NVMe SSD</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">CPU:</span>
                <span className="text-terminal-accent">32+ cores (Xeon/EPYC)</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Software Requirements */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-terminal-accent mb-4 flex items-center gap-2">
          💻 Software Requirements
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold text-terminal-secondary mb-3">Operating System</h4>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-terminal-primary">✓</span>
                <span className="text-gray-300">Ubuntu 22.04 LTS (recommended)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-terminal-primary">✓</span>
                <span className="text-gray-300">Ubuntu 20.04 LTS</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-gray-500">○</span>
                <span className="text-gray-500">Other Linux distros (manual setup)</span>
              </div>
            </div>
          </div>
          <div>
            <h4 className="font-semibold text-terminal-secondary mb-3">Dependencies</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-300">Python:</span>
                <span className="text-terminal-primary">3.8 - 3.11 (3.10 recommended)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">CUDA:</span>
                <span className="text-terminal-primary">12.1+ (auto-installed)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">PyTorch:</span>
                <span className="text-terminal-primary">2.3.0+ (auto-installed)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Docker:</span>
                <span className="text-terminal-primary">Optional (for containers)</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Verification Command */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-terminal-accent mb-4 flex items-center gap-2">
          🔍 System Verification
        </h3>
        <p className="text-gray-300 mb-4">
          Run this command to verify your system meets the requirements:
        </p>
        <div className="code-block">
          <div className="flex items-center justify-between mb-2">
            <span className="text-terminal-secondary">$ </span>
            <button 
              onClick={() => navigator.clipboard.writeText('curl -fsSL https://raw.githubusercontent.com/twobitapps/480b-setup/main/scripts/system_check.sh | bash')}
              className="text-xs bg-gray-700 px-2 py-1 rounded hover:bg-gray-600 transition-colors"
            >
              Copy
            </button>
          </div>
          <code className="text-terminal-primary break-all">
            curl -fsSL https://raw.githubusercontent.com/twobitapps/480b-setup/main/scripts/system_check.sh | bash
          </code>
        </div>
        <div className="mt-4 p-4 bg-gray-800 rounded">
          <h4 className="font-semibold text-terminal-secondary mb-2">Example Output:</h4>
          <div className="text-xs space-y-1 font-mono">
            <div className="text-terminal-primary">✓ Ubuntu 22.04 (✓ &gt;= 20.04)</div>
            <div className="text-terminal-primary">✓ CPU cores: 32 (✓ &gt;= 16 recommended)</div>
            <div className="text-terminal-primary">✓ System RAM: 128GB (✓ &gt;= 64GB recommended)</div>
            <div className="text-terminal-primary">✓ Available space: 800GB (✓ &gt;= 500GB required)</div>
            <div className="text-terminal-primary">✓ GPU: NVIDIA H100 80GB HBM3</div>
            <div className="text-terminal-primary">✓ GPU Memory: 81920MB (✓ &gt;= 80000MB recommended)</div>
            <div className="text-terminal-accent">🎉 System ready for Qwen3-Coder-480B installation!</div>
          </div>
        </div>
      </div>

      {/* Network Requirements */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-terminal-accent mb-4 flex items-center gap-2">
          🌐 Network Requirements
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold text-terminal-secondary mb-3">Download Requirements</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-300">Model Size:</span>
                <span className="text-terminal-primary">~450GB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Dependencies:</span>
                <span className="text-terminal-primary">~5GB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Total Download:</span>
                <span className="text-terminal-accent">~455GB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Bandwidth:</span>
                <span className="text-terminal-primary">100+ Mbps recommended</span>
              </div>
            </div>
          </div>
          <div>
            <h4 className="font-semibold text-terminal-secondary mb-3">Download Time Estimates</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-300">1 Gbps:</span>
                <span className="text-terminal-primary">~1 hour</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">500 Mbps:</span>
                <span className="text-terminal-primary">~2 hours</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">100 Mbps:</span>
                <span className="text-terminal-primary">~10 hours</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">50 Mbps:</span>
                <span className="text-terminal-warning">~20 hours</span>
              </div>
            </div>
          </div>
        </div>
        <div className="mt-4 p-3 bg-yellow-900/20 border border-yellow-600/30 rounded">
          <p className="text-yellow-400 text-sm">
            💡 <strong>Tip:</strong> The installer supports resume functionality, so interrupted downloads can be continued.
          </p>
        </div>
      </div>
    </div>
  )
}