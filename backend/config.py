import os
import torch
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# --- Device ---
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# --- StyleGAN3 ---
# AFHQv2 512px model — fits in MPS (~2GB vs 7GB for 1024px)
STYLEGAN3_PKL = MODEL_DIR / "stylegan3-t-afhqv2-512x512.pkl"
STYLEGAN3_RESOLUTION = 512
STYLEGAN3_TEXTURE_SIZE = 256
# Download URL (NVIDIA official)
STYLEGAN3_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-afhqv2-512x512.pkl"

# --- Video ---
VIDEO_FPS = 30
VIDEO_CODEC = "libx264"
VIDEO_CRF = 18

# --- Audio ---
AUDIO_SR = 48000         # 7.1.4 standard sample rate
AUDIO_HOP_LENGTH = 1024  # ~21ms at 48kHz
NUM_CHANNELS = 12

CHANNEL_NAMES = [
    "L", "R", "C", "LFE",
    "Ls", "Rs", "Lrs", "Rrs",
    "Ltf", "Rtf", "Ltr", "Rtr",
]
