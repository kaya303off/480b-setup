import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Qwen3-Coder-480B Setup Guide',
  description: 'Complete automated installation guide for Qwen3-Coder-480B-A35B-Instruct model on Ubuntu with NVIDIA GPUs',
  keywords: 'Qwen, 480B, AI, LLM, Installation, NVIDIA, GPU, Ubuntu, Setup',
  authors: [{ name: 'Claude Code' }],
  viewport: 'width=device-width, initial-scale=1',
  robots: 'index, follow',
  openGraph: {
    title: 'Qwen3-Coder-480B Setup Guide',
    description: 'Complete automated installation guide for Qwen3-Coder-480B-A35B-Instruct',
    url: 'https://480b-setup.vercel.app',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Qwen3-Coder-480B Setup Guide',
    description: 'Complete automated installation guide for Qwen3-Coder-480B-A35B-Instruct',
  }
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}