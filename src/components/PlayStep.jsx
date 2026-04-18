import { useState, useEffect, useRef, useCallback } from 'react'
import { motion } from 'framer-motion'
import ScreenLayout from './ScreenLayout'

const CHANNELS = ['L', 'R', 'C', 'LFE', 'Ls', 'Rs', 'Lrs', 'Rrs', 'Ltf', 'Rtf', 'Ltr', 'Rtr']
const CHANNEL_COLORS = {
  L: '#3b82f6', R: '#ef4444', C: '#22c55e', LFE: '#f59e0b',
  Ls: '#6366f1', Rs: '#ec4899', Lrs: '#8b5cf6', Rrs: '#f43f5e',
  Ltf: '#06b6d4', Rtf: '#f97316', Ltr: '#14b8a6', Rtr: '#e879f9',
}

const API_BASE = 'http://localhost:8000'

const PlayStep = ({ imageUrl, audioUrl, videoUrl, channelData, onReset }) => {
  const [isPlaying, setIsPlaying] = useState(false)
  const [levels, setLevels] = useState(() => CHANNELS.reduce((acc, ch) => ({ ...acc, [ch]: 0 }), {}))
  const audioRef = useRef(null)
  const videoRef = useRef(null)
  const analyserRef = useRef(null)
  const animFrameRef = useRef(null)
  const audioContextRef = useRef(null)

  // Determine if we have a backend-generated video
  const hasVideo = !!videoUrl
  const fullVideoUrl = hasVideo
    ? (videoUrl.startsWith('http') ? videoUrl : `${API_BASE}${videoUrl}`)
    : null

  const setupAudio = useCallback(() => {
    if (audioContextRef.current) return
    const ctx = new (window.AudioContext || window.webkitAudioContext)()
    audioContextRef.current = ctx

    // Use video element as source if we have a video, otherwise audio element
    const mediaElement = hasVideo ? videoRef.current : audioRef.current
    if (!mediaElement) return

    const source = ctx.createMediaElementSource(mediaElement)
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 256
    source.connect(analyser)
    analyser.connect(ctx.destination)
    analyserRef.current = analyser
  }, [hasVideo])

  const updateLevels = useCallback(() => {
    if (!analyserRef.current) return
    const data = new Uint8Array(analyserRef.current.frequencyBinCount)
    analyserRef.current.getByteFrequencyData(data)

    const binSize = Math.floor(data.length / CHANNELS.length)
    const newLevels = {}
    CHANNELS.forEach((ch, i) => {
      const start = i * binSize
      const slice = data.slice(start, start + binSize)
      const avg = slice.reduce((a, b) => a + b, 0) / slice.length / 255
      newLevels[ch] = avg
    })
    setLevels(newLevels)
    animFrameRef.current = requestAnimationFrame(updateLevels)
  }, [])

  const togglePlay = () => {
    const mediaElement = hasVideo ? videoRef.current : audioRef.current
    if (!mediaElement) return
    setupAudio()
    if (isPlaying) {
      mediaElement.pause()
      cancelAnimationFrame(animFrameRef.current)
    } else {
      mediaElement.play()
      updateLevels()
    }
    setIsPlaying(!isPlaying)
  }

  useEffect(() => {
    return () => {
      cancelAnimationFrame(animFrameRef.current)
      audioContextRef.current?.close()
    }
  }, [])

  const [showJson, setShowJson] = useState(false)

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex flex-col gap-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">Playback</h2>
          <p className="text-sm text-gray-400">
            {hasVideo ? 'Generated spatial video' : 'Real-time spatial audio visualization'}
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setShowJson(!showJson)}
            className="px-4 py-2 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 text-sm transition-all"
          >
            {showJson ? 'Hide' : 'Show'} JSON
          </button>
          <button
            onClick={onReset}
            className="px-4 py-2 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 text-sm transition-all"
          >
            New Project
          </button>
        </div>
      </div>

      {/* Video or Screen Preview */}
      <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800">
        {hasVideo ? (
          <video
            ref={videoRef}
            src={fullVideoUrl}
            className="w-full rounded-lg"
            crossOrigin="anonymous"
          />
        ) : (
          <ScreenLayout imageUrl={imageUrl} />
        )}
      </div>

      {/* Audio Controls (only when no video) */}
      {!hasVideo && <audio ref={audioRef} src={audioUrl} />}

      <div className="flex items-center gap-4 bg-gray-900/50 rounded-xl p-4 border border-gray-800">
        <button
          onClick={togglePlay}
          className="w-12 h-12 rounded-full bg-gradient-to-r from-blue-600 to-purple-600 flex items-center justify-center hover:shadow-lg hover:shadow-blue-500/30 transition-all shrink-0"
        >
          {isPlaying ? (
            <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          ) : (
            <svg className="w-5 h-5 text-white ml-0.5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>

        {/* Level Meters */}
        <div className="flex-1 grid grid-cols-12 gap-1 h-16">
          {CHANNELS.map(ch => (
            <div key={ch} className="flex flex-col items-center gap-1">
              <div className="flex-1 w-full bg-gray-800 rounded-sm overflow-hidden relative">
                <motion.div
                  className="absolute bottom-0 w-full rounded-sm"
                  animate={{ height: `${levels[ch] * 100}%` }}
                  transition={{ duration: 0.05 }}
                  style={{ backgroundColor: CHANNEL_COLORS[ch] }}
                />
              </div>
              <span className="text-[8px] font-mono text-gray-500">{ch}</span>
            </div>
          ))}
        </div>
      </div>

      {/* JSON Output */}
      {showJson && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="bg-gray-900 rounded-xl p-4 border border-gray-800 overflow-hidden"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono text-gray-500">channel_positions.json</span>
            <button
              onClick={() => navigator.clipboard.writeText(JSON.stringify(channelData, null, 2))}
              className="text-xs text-blue-400 hover:text-blue-300"
            >
              Copy
            </button>
          </div>
          <pre className="text-xs font-mono text-green-400 overflow-auto max-h-60">
            {JSON.stringify(channelData, null, 2)}
          </pre>
        </motion.div>
      )}
    </motion.div>
  )
}

export default PlayStep
