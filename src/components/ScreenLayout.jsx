import { forwardRef } from 'react'

const ScreenPanel = ({ label, imageUrl, side, className = '' }) => {
  const clipStyle = side === 'left'
    ? { objectPosition: 'left center', objectFit: 'cover' }
    : side === 'right'
    ? { objectPosition: 'right center', objectFit: 'cover' }
    : { objectFit: 'cover' }

  return (
    <div className={`relative overflow-hidden border border-gray-700/50 rounded-lg bg-gray-900 ${className}`}>
      {imageUrl && (
        <img
          src={imageUrl}
          alt={label}
          className="absolute inset-0 w-full h-full"
          style={clipStyle}
        />
      )}
      <div className="absolute inset-0 bg-black/30 flex items-end justify-center pb-2">
        <span className="text-xs font-mono text-white/60 bg-black/40 px-2 py-0.5 rounded">{label}</span>
      </div>
    </div>
  )
}

const ScreenLayout = forwardRef(({ imageUrl }, ref) => {
  return (
    <div ref={ref} className="relative flex h-[520px] gap-1.5 select-none">
      {/* Left Screens */}
      <div className="flex w-[15%] gap-1">
        <ScreenPanel label="L2" imageUrl={imageUrl} side="left" className="w-1/2 h-full" />
        <ScreenPanel label="L1" imageUrl={imageUrl} side="left" className="w-1/2 h-full" />
      </div>

      {/* Center Screen */}
      <div className="flex-1 relative overflow-hidden border-2 border-blue-500/40 rounded-lg bg-gray-900">
        {imageUrl && (
          <img src={imageUrl} alt="Center" className="w-full h-full object-cover" />
        )}
        <div className="absolute inset-0 bg-blue-500/5 flex items-end justify-center pb-2">
          <span className="text-sm font-mono text-blue-400/70 bg-black/40 px-3 py-0.5 rounded">CENTER</span>
        </div>
      </div>

      {/* Right Screens */}
      <div className="flex w-[15%] gap-1">
        <ScreenPanel label="R1" imageUrl={imageUrl} side="right" className="w-1/2 h-full" />
        <ScreenPanel label="R2" imageUrl={imageUrl} side="right" className="w-1/2 h-full" />
      </div>
    </div>
  )
})

ScreenLayout.displayName = 'ScreenLayout'

export default ScreenLayout
