"""
12-channel (7.1.4) audio analysis using Librosa.
Extracts per-channel RMS envelopes and onset strength for spatial modulation.
"""

import subprocess
import tempfile

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

from backend.config import AUDIO_SR, AUDIO_HOP_LENGTH, NUM_CHANNELS, CHANNEL_NAMES


def _extract_audio_to_wav(input_path: str | Path) -> str:
    """Extract audio from a container (mp4, mkv, etc.) to a temp WAV file using ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path), "-vn", "-acodec", "pcm_f32le", tmp.name],
        check=True, capture_output=True,
    )
    return tmp.name


def load_multichannel(audio_path: str | Path, target_sr: int = AUDIO_SR) -> np.ndarray:
    """
    Load a multi-channel audio file (WAV, FLAC, MP4, MKV, etc.).
    Returns shape (num_channels, num_samples).
    If the file has fewer channels than 12, remaining channels are zero-padded.
    """
    audio_path = Path(audio_path)
    # For container formats that soundfile can't read, extract via ffmpeg
    if audio_path.suffix.lower() in ('.mp4', '.mkv', '.mov', '.m4a', '.aac', '.ogg', '.webm'):
        wav_path = _extract_audio_to_wav(audio_path)
        data, sr = sf.read(wav_path, always_2d=True)
        Path(wav_path).unlink(missing_ok=True)
    else:
        data, sr = sf.read(str(audio_path), always_2d=True)  # (samples, channels)
    data = data.T  # (channels, samples)

    # Resample if needed
    if sr != target_sr:
        resampled = []
        for ch in range(data.shape[0]):
            resampled.append(librosa.resample(data[ch], orig_sr=sr, target_sr=target_sr))
        data = np.array(resampled)

    # Pad to 12 channels if needed
    if data.shape[0] < NUM_CHANNELS:
        pad = np.zeros((NUM_CHANNELS - data.shape[0], data.shape[1]), dtype=data.dtype)
        data = np.concatenate([data, pad], axis=0)
    elif data.shape[0] > NUM_CHANNELS:
        data = data[:NUM_CHANNELS]

    return data


def extract_rms(
    audio: np.ndarray,
    hop_length: int = AUDIO_HOP_LENGTH,
) -> dict[str, np.ndarray]:
    """
    Extract per-channel RMS envelope.
    Returns dict mapping channel name -> RMS array (num_frames,).
    """
    rms_data = {}
    for i, name in enumerate(CHANNEL_NAMES):
        rms = librosa.feature.rms(y=audio[i], hop_length=hop_length)[0]
        # Normalize to [0, 1]
        peak = rms.max()
        if peak > 0:
            rms = rms / peak
        rms_data[name] = rms.astype(np.float32)
    return rms_data


def extract_onsets(
    audio: np.ndarray,
    sr: int = AUDIO_SR,
    hop_length: int = AUDIO_HOP_LENGTH,
) -> dict[str, np.ndarray]:
    """
    Extract per-channel onset strength envelope.
    Returns dict mapping channel name -> onset strength array (num_frames,).
    """
    onset_data = {}
    for i, name in enumerate(CHANNEL_NAMES):
        onset_env = librosa.onset.onset_strength(
            y=audio[i], sr=sr, hop_length=hop_length
        )
        peak = onset_env.max()
        if peak > 0:
            onset_env = onset_env / peak
        onset_data[name] = onset_env.astype(np.float32)
    return onset_data


def analyze_audio(audio_path: str | Path) -> dict:
    """
    Full analysis pipeline.
    Returns a dict with per-channel RMS and onset data,
    plus metadata (duration, num_frames, sr).
    """
    audio = load_multichannel(audio_path)
    num_samples = audio.shape[1]
    duration = num_samples / AUDIO_SR
    num_frames = 1 + num_samples // AUDIO_HOP_LENGTH

    rms = extract_rms(audio)
    onsets = extract_onsets(audio)

    return {
        "duration": float(duration),
        "num_frames": int(num_frames),
        "sr": AUDIO_SR,
        "hop_length": AUDIO_HOP_LENGTH,
        "channels": {
            name: {
                "rms": rms[name].tolist(),
                "onset": onsets[name].tolist(),
            }
            for name in CHANNEL_NAMES
        },
    }
