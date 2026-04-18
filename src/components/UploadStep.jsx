import { useRef, useState } from 'react'
import { motion } from 'framer-motion'

const UploadStep = ({ onComplete }) => {
  const imageInputRef = useRef(null)
  const audioInputRef = useRef(null)
  const [useReference, setUseReference] = useState(false)
  const [imageSelected, setImageSelected] = useState(false)
  const [audioSelected, setAudioSelected] = useState(false)
  const imageFileRef = useRef(null)
  const audioFileRef = useRef(null)

  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      imageFileRef.current = file
      setImageSelected(true)
      tryComplete(file, audioFileRef.current)
    }
  }

  const handleAudioChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      audioFileRef.current = file
      setAudioSelected(true)
      if (useReference) {
        tryComplete(imageFileRef.current, file)
      } else {
        completeAudioOnly(file)
      }
    }
  }

  const completeAudioOnly = (audFile) => {
    const audioUrl = URL.createObjectURL(audFile)
    setTimeout(() => onComplete({
      imageUrl: null,
      audioUrl,
      imageFile: null,
      audioFile: audFile,
      useReference: false,
    }), 400)
  }

  const tryComplete = (imgFile, audFile) => {
    if (imgFile && audFile) {
      const imageUrl = URL.createObjectURL(imgFile)
      const audioUrl = URL.createObjectURL(audFile)
      setTimeout(() => onComplete({
        imageUrl,
        audioUrl,
        imageFile: imgFile,
        audioFile: audFile,
        useReference: true,
      }), 400)
    }
  }

  const handleToggleReference = () => {
    const next = !useReference
    setUseReference(next)
    // If turning off reference and audio is already selected, proceed
    if (!next && audioFileRef.current) {
      completeAudioOnly(audioFileRef.current)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center justify-center min-h-[70vh] gap-8"
    >
      <div className="text-center mb-4">
        <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          7.1.4 Spatial Audio Visualizer
        </h1>
        <p className="text-gray-400 text-lg">Upload your 7.1.4 channel audio to visualize</p>
      </div>

      <div className="flex flex-col items-center gap-6">
        {/* Reference Image Toggle */}
        <label className="flex items-center gap-3 cursor-pointer select-none">
          <div
            onClick={handleToggleReference}
            className={`relative w-11 h-6 rounded-full transition-colors ${
              useReference ? 'bg-blue-600' : 'bg-gray-700'
            }`}
          >
            <div className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
              useReference ? 'translate-x-5' : 'translate-x-0'
            }`} />
          </div>
          <span className="text-sm text-gray-400">Use reference image</span>
        </label>

        <div className="flex gap-6">
          {/* Image Upload (conditional) */}
          {useReference && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => imageInputRef.current.click()}
              className="w-72 h-48 border-2 border-dashed border-gray-600 rounded-xl flex flex-col items-center justify-center cursor-pointer hover:border-blue-500 hover:bg-blue-500/5 transition-all"
            >
              <input ref={imageInputRef} type="file" accept="image/*" className="hidden" onChange={handleImageChange} />
              <svg className="w-12 h-12 text-gray-500 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <span className={`text-sm font-medium ${imageSelected ? 'text-green-400' : 'text-gray-400'}`}>
                {imageSelected ? '✓ Image Selected' : 'Reference Image'}
              </span>
              <span className="text-gray-600 text-xs mt-1">PNG, JPG, WebP</span>
            </motion.div>
          )}

          {/* Audio Upload */}
          <motion.div
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => audioInputRef.current.click()}
            className="w-72 h-48 border-2 border-dashed border-gray-600 rounded-xl flex flex-col items-center justify-center cursor-pointer hover:border-purple-500 hover:bg-purple-500/5 transition-all"
          >
            <input ref={audioInputRef} type="file" accept="audio/*,video/*" className="hidden" onChange={handleAudioChange} />
            <svg className="w-12 h-12 text-gray-500 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
            </svg>
            <span className={`text-sm font-medium ${audioSelected ? 'text-green-400' : 'text-gray-400'}`}>
              {audioSelected ? '✓ Audio Selected' : '7.1.4 Audio File'}
            </span>
            <span className="text-gray-600 text-xs mt-1">WAV, FLAC, MP4, MKV</span>
          </motion.div>
        </div>
      </div>
    </motion.div>
  )
}

export default UploadStep
