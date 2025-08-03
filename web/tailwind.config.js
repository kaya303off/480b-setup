/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        terminal: {
          bg: '#0a0a0a',
          primary: '#00ff00',
          secondary: '#ffff00',
          accent: '#00ffff',
          error: '#ff0000',
          warning: '#ff9500',
        },
      },
      fontFamily: {
        mono: ['Monaco', 'Menlo', 'Consolas', 'Courier New', 'monospace'],
      },
    },
  },
  plugins: [],
}