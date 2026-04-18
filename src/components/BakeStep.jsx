import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'

const BAKE_MESSAGES = {
  stylegan: [
    'Analyzing spatial channel layout...',
    'Extracting RMS envelope per channel...',
    'Computing onset detection (Librosa)...',
    'Building Spatial Latent Mask...',
    'Initializing StyleGAN3 W-space vectors...',
    'Inverting reference image into W-space...',
    'Generating weight maps from coordinates...',
    'U-Net denoising pass 1/4...',
    'U-Net denoising pass 2/4...',
    'U-Net denoising pass 3/4...',
    'U-Net denoising pass 4/4...',
    'Warping latent vectors with audio energy...',
    'Applying spatial modulation to feature maps...',
    'Rendering video frames (MPS accelerated)...',
    'Encoding with FFmpeg (H.264)...',
    'Finalizing output...',
  ],
  physics: [
    'Preparing 5-screen layout...',
    'Analyzing 12-channel RMS envelopes...',
    'Initializing particle system...',
    'Emitting per-channel bursts...',
    'Integrating velocities with drag...',
    'Compositing additive bloom...',
    'Rendering frames...',
    'Encoding with FFmpeg (H.264)...',
    'Finalizing output...',
  ],
}

const API_BASE = 'http://localhost:8000'

const ENDPOINTS = {
  stylegan: '/api/bake',
  physics: '/api/bake_physics',
}

const BakeStep = ({ channelData, imageFile, audioFile, useReference = false, onComplete, useBackend = false, mode = 'stylegan' }) => {
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState(BAKE_MESSAGES[mode][0])
  const intervalRef = useRef(null)

  // --- Simulation mode (no backend) ---
  useEffect(() => {
    if (useBackend) return

    intervalRef.current = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(intervalRef.current)
          setTimeout(() => onComplete(null), 800)
          return 100
        }
        const increment = Math.random() * 1.5 + 0.3
        const next = Math.min(100, prev + increment)

        // Update message
        const msgs = BAKE_MESSAGES[mode]
        const idx = Math.min(
          Math.floor((next / 100) * msgs.length),
          msgs.length - 1
        )
        setMessage(msgs[idx])

        return next
      })
    }, 120)

    return () => clearInterval(intervalRef.current)
  }, [onComplete, useBackend, mode])

  // --- Real backend mode (SSE) ---
  useEffect(() => {
    if (!useBackend || !audioFile) return

    const formData = new FormData()
    if (useReference && imageFile) {
      formData.append('image', imageFile)
    }
    formData.append('audio', audioFile)
    formData.append('channel_positions', JSON.stringify(channelData))

    const controller = new AbortController()

    ;(async () => {
      try {
        const endpoint = ENDPOINTS[mode] || ENDPOINTS.stylegan
        const response = await fetch(`${API_BASE}${endpoint}`, {
          method: 'POST',
          body: formData,
          signal: controller.signal,
        })

        if (!response.ok) {
          const errText = await response.text()
          setMessage(`Server error (${response.status}): ${errText}`)
          return
        }

        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() // keep incomplete line in buffer

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            try {
              const data = JSON.parse(line.slice(6))

              if (data.status === 'processing') {
                setProgress(data.progress * 100)
                setMessage(data.message)
              } else if (data.status === 'complete') {
                setProgress(100)
                setMessage('Complete!')
                setTimeout(() => onComplete(data.video_url), 800)
              } else if (data.status === 'error') {
                setMessage(`Error: ${data.message}`)
              }
            } catch {
              // skip malformed JSON chunks
            }
          }
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          setMessage(`Connection error: ${err.message}. Is the backend running on ${API_BASE}?`)
        }
      }
    })()

    return () => controller.abort()
  }, [useBackend, imageFile, audioFile, channelData, onComplete, mode, useReference])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex flex-col items-center justify-center min-h-[60vh] gap-8"
    >
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-2">Generating Spatial Video</h2>
        <p className="text-gray-400">
          {mode === 'physics'
            ? 'Fluid particle field · RMS-driven physics'
            : 'StyleGAN3 · audio-reactive latent warping'}
        </p>
      </div>

      {/* Progress Ring */}
      <div className="relative w-48 h-48">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="44" fill="none" stroke="#1f2937" strokeWidth="6" />
          <circle
            cx="50" cy="50" r="44"
            fill="none"
            stroke="url(#gradient)"
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={`${progress * 2.76} 276`}
            className="transition-all duration-200"
          />
          <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#3b82f6" />
              <stop offset="100%" stopColor="#a855f7" />
            </linearGradient>
          </defs>
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-bold text-white">{Math.floor(progress)}%</span>
        </div>
      </div>

      {/* Status Message */}
      <motion.div
        key={message}
        initial={{ opacity: 0, y: 5 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-sm text-blue-400 font-mono bg-gray-800/50 px-4 py-2 rounded-lg border border-gray-700/50"
      >
        {message}
      </motion.div>

      {/* Progress Bar */}
      <div className="w-96 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <motion.div
          className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Channel Data Preview */}
      <div className="text-xs text-gray-600 font-mono max-w-lg text-center">
        Channels: {Object.keys(channelData).join(', ')}
      </div>
    </motion.div>
  )
}

export default BakeStep
