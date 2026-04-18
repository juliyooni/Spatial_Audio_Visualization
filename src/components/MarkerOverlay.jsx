import { motion } from 'framer-motion'

const CHANNEL_COLORS = {
  L: '#3b82f6',
  R: '#ef4444',
  C: '#22c55e',
  LFE: '#f59e0b',
  Ls: '#6366f1',
  Rs: '#ec4899',
  Lrs: '#8b5cf6',
  Rrs: '#f43f5e',
  Ltf: '#06b6d4',
  Rtf: '#f97316',
  Ltr: '#14b8a6',
  Rtr: '#e879f9',
}

const MarkerOverlay = ({ markers, onDrag, containerRef, isFlat }) => {
  const handleDragEnd = (id, event) => {
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const el = event.target.getBoundingClientRect()
    const centerX = el.left + el.width / 2
    const centerY = el.top + el.height / 2
    const normX = Math.max(0, Math.min(1, (centerX - rect.left) / rect.width))
    const normY = Math.max(0, Math.min(1, (centerY - rect.top) / rect.height))
    onDrag(id, normX, normY)
  }

  return (
    <div className="absolute inset-0 pointer-events-none z-20">
      {markers.map((m) => (
        <motion.div
          key={m.id}
          drag={!isFlat}
          dragMomentum={false}
          dragConstraints={containerRef}
          onDragEnd={(e, info) => handleDragEnd(m.id, e)}
          animate={isFlat ? {
            left: '50%',
            top: '50%',
            x: '-50%',
            y: '-50%',
          } : undefined}
          transition={isFlat ? { type: 'spring', stiffness: 200, damping: 20 } : undefined}
          className="pointer-events-auto absolute flex flex-col items-center cursor-move select-none"
          style={!isFlat ? { left: `${m.x * 100}%`, top: `${m.y * 100}%`, transform: 'translate(-50%, -50%)' } : undefined}
        >
          <div
            className="w-9 h-9 rounded-full flex items-center justify-center text-[10px] font-bold text-white shadow-lg border-2 border-white/60 backdrop-blur-sm"
            style={{ backgroundColor: CHANNEL_COLORS[m.id] || '#6366f1' }}
          >
            {m.id}
          </div>
          <div className="text-[9px] text-gray-400 mt-0.5 whitespace-nowrap">{m.label}</div>
        </motion.div>
      ))}
    </div>
  )
}

export default MarkerOverlay
