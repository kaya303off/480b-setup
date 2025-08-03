export default function Footer() {
  return (
    <footer className="mt-12 text-center">
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <div className="grid md:grid-cols-3 gap-6 mb-6">
          <div>
            <h3 className="font-semibold text-terminal-secondary mb-2">Resources</h3>
            <div className="space-y-1 text-sm">
              <a 
                href="https://github.com/twobitapps/480b-setup"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors"
              >
                GitHub Repository
              </a>
              <a 
                href="https://github.com/twobitapps/480b-setup/blob/main/docs/TROUBLESHOOTING.md"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors"
              >
                Troubleshooting Guide
              </a>
              <a 
                href="https://github.com/twobitapps/480b-setup/releases"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors"
              >
                Download Releases
              </a>
            </div>
          </div>
          
          <div>
            <h3 className="font-semibold text-terminal-secondary mb-2">Community</h3>
            <div className="space-y-1 text-sm">
              <a 
                href="https://github.com/twobitapps/480b-setup/discussions"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors"
              >
                Discussions
              </a>
              <a 
                href="https://github.com/twobitapps/480b-setup/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors"
              >
                Report Issues
              </a>
              <a 
                href="https://github.com/twobitapps/480b-setup/wiki"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors"
              >
                Community Wiki
              </a>
            </div>
          </div>
          
          <div>
            <h3 className="font-semibold text-terminal-secondary mb-2">Related</h3>
            <div className="space-y-1 text-sm">
              <a 
                href="https://github.com/QwenLM/Qwen2.5-Coder"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors"
              >
                Official Qwen Repo
              </a>
              <a 
                href="https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors"
              >
                Hugging Face Model
              </a>
              <a 
                href="https://hello-world-vercel-opal.vercel.app"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-terminal-primary hover:text-terminal-accent transition-colors"
              >
                Model Comparison Demo
              </a>
            </div>
          </div>
        </div>
        
        <div className="border-t border-gray-700 pt-4">
          <p className="text-gray-400 text-sm mb-2">
            Professional installation suite for Qwen3-Coder-480B-A35B-Instruct
          </p>
          <p className="text-xs text-gray-500">
            🤖 Generated with{' '}
            <a 
              href="https://claude.ai/code" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-terminal-accent hover:text-terminal-secondary transition-colors"
            >
              Claude Code
            </a>
            {' • '}
            Co-Authored-By: Claude &lt;noreply@anthropic.com&gt;
          </p>
        </div>
      </div>
    </footer>
  )
}