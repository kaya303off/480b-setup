'use client'

import { useState, useEffect } from 'react'
import TerminalHeader from '@/components/TerminalHeader'
import InstallationSteps from '@/components/InstallationSteps'
import SystemRequirements from '@/components/SystemRequirements'
import QuickStart from '@/components/QuickStart'
import Documentation from '@/components/Documentation'
import Footer from '@/components/Footer'

export default function Home() {
  const [currentTime, setCurrentTime] = useState('')
  const [activeSection, setActiveSection] = useState('quick-start')

  useEffect(() => {
    const updateTime = () => {
      setCurrentTime(new Date().toISOString().slice(0, 19) + 'Z')
    }
    updateTime()
    const interval = setInterval(updateTime, 1000)
    return () => clearInterval(interval)
  }, [])

  const sections = [
    { id: 'quick-start', label: 'Quick Start', icon: '🚀' },
    { id: 'requirements', label: 'Requirements', icon: '📋' },
    { id: 'installation', label: 'Installation', icon: '⚙️' },
    { id: 'documentation', label: 'Documentation', icon: '📚' },
  ]

  const renderSection = () => {
    switch (activeSection) {
      case 'quick-start':
        return <QuickStart />
      case 'requirements':
        return <SystemRequirements />
      case 'installation':
        return <InstallationSteps />
      case 'documentation':
        return <Documentation />
      default:
        return <QuickStart />
    }
  }

  return (
    <div className="min-h-screen bg-terminal-bg text-terminal-primary font-mono">
      {/* Main Terminal Window */}
      <div className="container mx-auto p-4 max-w-6xl">
        <div className="terminal-window">
          <TerminalHeader currentTime={currentTime} />
          
          {/* ASCII Art Header */}
          <div className="terminal-content">
            <div className="text-center mb-8">
              <pre className="text-xs md:text-sm text-terminal-accent whitespace-pre-wrap">
{`
 ██████╗ ██╗    ██╗███████╗███╗   ██╗    ██╗  ██╗ █████╗  ██████╗ ██████╗ 
██╔═══██╗██║    ██║██╔════╝████╗  ██║    ██║  ██║██╔══██╗██╔═══██╗██╔══██╗
██║   ██║██║ █╗ ██║█████╗  ██╔██╗ ██║    ███████║╚█████╔╝██║   ██║██████╔╝
██║▄▄ ██║██║███╗██║██╔══╝  ██║╚██╗██║    ╚════██║██╔══██╗██║   ██║██╔══██╗
╚██████╔╝╚███╔███╔╝███████╗██║ ╚████║         ██║╚█████╔╝╚██████╔╝██████╔╝
 ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝         ╚═╝ ╚════╝  ╚═════╝ ╚═════╝ 
                                                                            
              Qwen3-Coder-480B-A35B-Instruct Setup Guide
                        Professional Installation Suite
`}
              </pre>
              <div className="text-terminal-secondary text-sm mt-4">
                <p>Complete automated installation for 480B parameter language model</p>
                <p className="text-terminal-accent">Ubuntu • NVIDIA GPU • CUDA • Docker Ready</p>
              </div>
            </div>

            {/* Navigation */}
            <div className="flex flex-wrap justify-center gap-2 mb-8 border-b border-gray-700 pb-4">
              {sections.map((section) => (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`px-4 py-2 rounded transition-all duration-200 flex items-center gap-2 ${
                    activeSection === section.id
                      ? 'bg-terminal-primary text-black glow'
                      : 'bg-gray-800 text-terminal-primary hover:bg-gray-700'
                  }`}
                >
                  <span>{section.icon}</span>
                  <span className="hidden sm:inline">{section.label}</span>
                </button>
              ))}
            </div>

            {/* Content Section */}
            <div className="transition-all duration-300">
              {renderSection()}
            </div>
          </div>
        </div>

        <Footer />
      </div>
    </div>
  )
}