import { useState, useRef, useCallback } from 'react'
import { motion } from 'framer-motion'
import ScreenLayout from './ScreenLayout'
import MarkerOverlay from './MarkerOverlay'

const DEFAULT_MARKERS_714 = [
  { id: 'L',   label: 'Left',           x: 0.25, y: 0.45 },
  { id: 'R',   label: 'Right',          x: 0.75, y: 0.45 },
  { id: 'C',   label: 'Center',         x: 0.50, y: 0.40 },
  { id: 'LFE', label: 'Sub',            x: 0.50, y: 0.70 },
  { id: 'Ls',  label: 'Left Surround',  x: 0.10, y: 0.50 },
  { id: 'Rs',  label: 'Right Surround', x: 0.90, y: 0.50 },
  { id: 'Lrs', label: 'Left Rear',      x: 0.08, y: 0.75 },
  { id: 'Rrs', label: 'Right Rear',     x: 0.92, y: 0.75 },
  { id: 'Ltf', label: 'Left Top Front', x: 0.30, y: 0.20 },
  { id: 'Rtf', label: 'Right Top Front',x: 0.70, y: 0.20 },
  { id: 'Ltr', label: 'Left Top Rear',  x: 0.20, y: 0.30 },
  { id: 'Rtr', label: 'Right Top Rear', x: 0.80, y: 0.30 },
]

const SetupStep = ({ imageUrl, onBake }) => {
  const [markers, setMarkers] = useState(DEFAULT_MARKERS_714)
  const [isFlat, setIsFlat] = useState(false)
  const containerRef = useRef(null)

  const handleDrag = useCallback((id, normX, normY) => {
    setMarkers(prev => prev.map(m =>
      m.id === id ? { ...m, x: normX, y: normY } : m
    ))
  }, [])

  const handleBake = () => {
    const channelData = {}
    markers.forEach(m => {
      channelData[m.id] = { x: parseFloat(m.x.toFixed(4)), y: parseFloat(m.y.toFixed(4)), label: m.label }
    })
    onBake(channelData)
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex flex-col gap-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">Channel Setup</h2>
          <p className="text-sm text-gray-400">Drag markers to set audio source positions</p>
        </div>
        <div className="flex items-center gap-3">
          {/* Mode Toggle */}
          <div className="flex bg-gray-800 rounded-lg p-0.5">
            <button
              onClick={() => setIsFlat(false)}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
                !isFlat ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
              }`}
            >
              7.1.4
            </button>
            <button
              onClick={() => setIsFlat(true)}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
                isFlat ? 'bg-gray-600 text-white' : 'text-gray-400 hover:text-white'
              }`}
            >
              Flat
            </button>
          </div>

          <button
            onClick={handleBake}
            className="px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg hover:from-blue-500 hover:to-purple-500 transition-all shadow-lg shadow-blue-500/20"
          >
            Bake
          </button>
        </div>
      </div>

      {/* Screen Layout with Markers */}
      <div className="relative bg-gray-900/50 rounded-xl p-4 border border-gray-800">
        <div ref={containerRef} className="relative">
          <ScreenLayout ref={containerRef} imageUrl={imageUrl} />
          <MarkerOverlay
            markers={markers}
            onDrag={handleDrag}
            containerRef={containerRef}
            isFlat={isFlat}
          />
        </div>
      </div>

      {/* Channel Info Panel */}
      <div className="grid grid-cols-6 gap-2">
        {markers.map(m => (
          <div key={m.id} className="bg-gray-800/50 rounded-lg p-2 text-center border border-gray-700/30">
            <div className="text-xs font-bold text-white">{m.id}</div>
            <div className="text-[10px] text-gray-500 mt-0.5">
              ({m.x.toFixed(2)}, {m.y.toFixed(2)})
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  )
}

export default SetupStep
