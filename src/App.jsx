import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import UploadStep from './components/UploadStep'
import SetupStep from './components/SetupStep'
import BakeStep from './components/BakeStep'
import PlayStep from './components/PlayStep'

const STEPS = ['Upload', 'Setup', 'Bake', 'Play']

// Set to true when backend server is running at localhost:8000
const USE_BACKEND = true

function App() {
  const [step, setStep] = useState(0)
  const [imageUrl, setImageUrl] = useState(null)
  const [audioUrl, setAudioUrl] = useState(null)
  const [imageFile, setImageFile] = useState(null)
  const [audioFile, setAudioFile] = useState(null)
  const [useReference, setUseReference] = useState(false)
  const [channelData, setChannelData] = useState(null)
  const [videoUrl, setVideoUrl] = useState(null)
  const [bakeMode, setBakeMode] = useState('physics') // 'physics' | 'stylegan'

  const handleUploadComplete = useCallback((data) => {
    setImageUrl(data.imageUrl)
    setAudioUrl(data.audioUrl)
    setImageFile(data.imageFile)
    setAudioFile(data.audioFile)
    setUseReference(data.useReference)
    setStep(1)
  }, [])

  const handleBake = useCallback((data) => {
    setChannelData(data)
    setStep(2)
  }, [])

  const handleBakeComplete = useCallback((generatedVideoUrl) => {
    // generatedVideoUrl is null in simulation mode, or "/api/video/xxx.mp4" from backend
    setVideoUrl(generatedVideoUrl)
    setStep(3)
  }, [])

  const handleReset = useCallback(() => {
    setStep(0)
    setImageUrl(null)
    setAudioUrl(null)
    setImageFile(null)
    setAudioFile(null)
    setUseReference(false)
    setChannelData(null)
    setVideoUrl(null)
  }, [])

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Top Bar */}
      <header className="border-b border-gray-800 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-xs font-bold">
            SA
          </div>
          <span className="font-semibold text-sm text-white">Spatial Audio Viz</span>
          <span className="text-xs text-gray-600 font-mono ml-2">7.1.4ch</span>

          {/* Bake mode toggle */}
          <div className="ml-4 flex items-center gap-1 bg-gray-900 border border-gray-800 rounded-full p-0.5">
            <button
              onClick={() => setBakeMode('physics')}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${
                bakeMode === 'physics'
                  ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
              title="Fluid particle field — per-channel RMS physics"
            >
              Particles
            </button>
            <button
              onClick={() => setBakeMode('stylegan')}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${
                bakeMode === 'stylegan'
                  ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
              title="StyleGAN3 latent warping"
            >
              StyleGAN
            </button>
          </div>
        </div>

        {/* Step Indicator */}
        <div className="flex items-center gap-1">
          {STEPS.map((s, i) => (
            <div key={s} className="flex items-center">
              <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition-all ${
                i === step
                  ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
                  : i < step
                  ? 'text-green-400'
                  : 'text-gray-600'
              }`}>
                {i < step ? (
                  <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                ) : (
                  <span className="w-4 text-center">{i + 1}</span>
                )}
                {s}
              </div>
              {i < STEPS.length - 1 && (
                <div className={`w-6 h-px mx-1 ${i < step ? 'bg-green-500/50' : 'bg-gray-800'}`} />
              )}
            </div>
          ))}
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-6">
        <AnimatePresence mode="wait">
          {step === 0 && (
            <UploadStep key="upload" onComplete={handleUploadComplete} />
          )}
          {step === 1 && (
            <SetupStep key="setup" imageUrl={imageUrl} onBake={handleBake} />
          )}
          {step === 2 && (
            <BakeStep
              key={`bake-${bakeMode}`}
              channelData={channelData}
              imageFile={imageFile}
              audioFile={audioFile}
              useReference={useReference}
              useBackend={USE_BACKEND}
              mode={bakeMode}
              onComplete={handleBakeComplete}
            />
          )}
          {step === 3 && (
            <PlayStep
              key="play"
              imageUrl={imageUrl}
              audioUrl={audioUrl}
              videoUrl={videoUrl}
              channelData={channelData}
              onReset={handleReset}
            />
          )}
        </AnimatePresence>
      </main>
    </div>
  )
}

export default App
