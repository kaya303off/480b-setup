export default function InstallationSteps() {
  const steps = [
    {
      number: 1,
      title: "System Verification",
      description: "Check system requirements and compatibility",
      command: "curl -fsSL https://raw.githubusercontent.com/twobitapps/480b-setup/main/scripts/system_check.sh | bash",
      time: "~30 seconds",
      status: "required"
    },
    {
      number: 2,
      title: "Download Installer",
      description: "Get the latest installation script",
      command: "curl -fsSL https://raw.githubusercontent.com/twobitapps/480b-setup/main/install.sh -o install.sh && chmod +x install.sh",
      time: "~5 seconds",
      status: "required"
    },
    {
      number: 3,
      title: "Run Installation",
      description: "Execute automated installation with progress tracking",
      command: "./install.sh",
      time: "30-90 minutes",
      status: "required"
    },
    {
      number: 4,
      title: "Verify Installation",
      description: "Test model loading and basic inference",
      command: "source ~/activate_qwen480b.sh && python test_inference.py",
      time: "2-5 minutes",
      status: "recommended"
    },
    {
      number: 5,
      title: "Performance Benchmark",
      description: "Run comprehensive performance tests",
      command: "python benchmark.py",
      time: "5-10 minutes",
      status: "optional"
    }
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'required': return 'text-red-400'
      case 'recommended': return 'text-yellow-400'
      case 'optional': return 'text-green-400'
      default: return 'text-gray-400'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'required': return '🔴'
      case 'recommended': return '🟡'
      case 'optional': return '🟢'
      default: return '⚪'
    }
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-terminal-secondary mb-2">
          ⚙️ Installation Steps
        </h2>
        <p className="text-terminal-primary">
          Follow these steps for a complete Qwen3-Coder-480B installation
        </p>
      </div>

      {/* Installation Timeline */}
      <div className="space-y-4">
        {steps.map((step, index) => (
          <div key={step.number} className="bg-gray-900 border border-gray-700 rounded-lg p-6">
            <div className="flex items-start gap-4">
              {/* Step Number */}
              <div className="flex-shrink-0 w-8 h-8 bg-terminal-primary text-black rounded-full flex items-center justify-center font-bold text-sm">
                {step.number}
              </div>
              
              {/* Step Content */}
              <div className="flex-1">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-lg font-semibold text-terminal-secondary">
                    {step.title}
                  </h3>
                  <div className="flex items-center gap-2 text-sm">
                    <span className={getStatusColor(step.status)}>
                      {getStatusIcon(step.status)} {step.status}
                    </span>
                    <span className="text-gray-400">• {step.time}</span>
                  </div>
                </div>
                
                <p className="text-gray-300 mb-3">{step.description}</p>
                
                <div className="code-block">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-terminal-secondary">$ </span>
                    <button 
                      onClick={() => navigator.clipboard.writeText(step.command)}
                      className="text-xs bg-gray-700 px-2 py-1 rounded hover:bg-gray-600 transition-colors"
                    >
                      Copy
                    </button>
                  </div>
                  <code className="text-terminal-primary break-all">
                    {step.command}
                  </code>
                </div>
              </div>
            </div>
            
            {/* Progress indicator */}
            {index < steps.length - 1 && (
              <div className="ml-4 mt-4 border-l-2 border-gray-700 h-4"></div>
            )}
          </div>
        ))}
      </div>

      {/* Installation Methods Comparison */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-terminal-accent mb-4 flex items-center gap-2">
          📊 Installation Methods Comparison
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left p-2 text-terminal-secondary">Method</th>
                <th className="text-left p-2 text-terminal-secondary">Time</th>
                <th className="text-left p-2 text-terminal-secondary">Difficulty</th>
                <th className="text-left p-2 text-terminal-secondary">Customization</th>
                <th className="text-left p-2 text-terminal-secondary">Recovery</th>
              </tr>
            </thead>
            <tbody className="text-gray-300">
              <tr className="border-b border-gray-800">
                <td className="p-2">
                  <span className="text-terminal-primary font-semibold">Automated Script</span>
                </td>
                <td className="p-2">30-90 min</td>
                <td className="p-2 text-green-400">Easy</td>
                <td className="p-2 text-yellow-400">Limited</td>
                <td className="p-2 text-green-400">Automatic</td>
              </tr>
              <tr className="border-b border-gray-800">
                <td className="p-2">
                  <span className="text-terminal-secondary font-semibold">Manual Install</span>
                </td>
                <td className="p-2">60-120 min</td>
                <td className="p-2 text-yellow-400">Medium</td>
                <td className="p-2 text-green-400">Full</td>
                <td className="p-2 text-yellow-400">Manual</td>
              </tr>
              <tr>
                <td className="p-2">
                  <span className="text-terminal-accent font-semibold">Docker</span>
                </td>
                <td className="p-2">45-75 min</td>
                <td className="p-2 text-green-400">Easy</td>
                <td className="p-2 text-yellow-400">Medium</td>
                <td className="p-2 text-green-400">Rebuild</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Troubleshooting Quick Links */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-terminal-accent mb-4 flex items-center gap-2">
          🛠️ Troubleshooting & Support
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-semibold text-terminal-secondary mb-2">Common Issues</h4>
            <ul className="space-y-1 text-sm text-gray-300">
              <li>• GPU memory errors → Check available VRAM</li>
              <li>• NumPy conflicts → Clean virtual environment</li>
              <li>• Download failures → Check network connectivity</li>
              <li>• CUDA errors → Verify driver compatibility</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-terminal-secondary mb-2">Get Help</h4>
            <div className="space-y-2">
              <a 
                href="https://github.com/twobitapps/480b-setup/blob/main/docs/TROUBLESHOOTING.md"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors text-sm"
              >
                📖 Troubleshooting Guide
              </a>
              <a 
                href="https://github.com/twobitapps/480b-setup/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors text-sm"
              >
                🐛 Report Issues
              </a>
              <a 
                href="https://github.com/twobitapps/480b-setup/discussions"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors text-sm"
              >
                💬 Community Support
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}