interface TerminalHeaderProps {
  currentTime: string
}

export default function TerminalHeader({ currentTime }: TerminalHeaderProps) {
  return (
    <div className="terminal-header">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="flex space-x-1">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
          </div>
          <span className="text-gray-300 text-sm font-mono">
            qwen-480b-setup@ubuntu:~/installation
          </span>
        </div>
        <div className="text-gray-400 text-xs font-mono">
          {currentTime}
        </div>
      </div>
    </div>
  )
}