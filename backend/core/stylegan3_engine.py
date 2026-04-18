"""
StyleGAN3 Engine — Texture Generator for Spatial Audio Visualization.

Role: Generate audio-reactive abstract textures via W-space modulation.
These textures are composited onto the user's reference image at
channel marker positions by the video_baker.

The engine does NOT produce the final video frame — it only generates
raw texture patches that pulse/morph with the audio.
"""

import gc
import pickle
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from PIL import Image as PILImage

from backend.config import DEVICE, STYLEGAN3_PKL, STYLEGAN3_RESOLUTION, STYLEGAN3_TEXTURE_SIZE, STYLEGAN3_URL
from backend.core.spatial_modulator import ChannelMarker


def _flush_mps():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
    gc.collect()


class StyleGAN3Engine:

    def __init__(self, pkl_path: Optional[Path] = None, device: torch.device = DEVICE):
        self.device = device
        self.pkl_path = pkl_path or STYLEGAN3_PKL
        self.G = None
        self.w_base = None

    def load_model(self):
        if self.G is not None:
            return
        if not self.pkl_path.exists():
            print(f"Downloading StyleGAN3 model to {self.pkl_path}...")
            import urllib.request
            self.pkl_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(STYLEGAN3_URL, str(self.pkl_path))
            print("Download complete.")

        with open(self.pkl_path, "rb") as f:
            data = pickle.load(f)
        self.G = data["G_ema"].eval()
        for p in self.G.parameters():
            p.requires_grad_(False)
        self.G = self.G.to(self.device)
        del data
        gc.collect()
        _flush_mps()

    def compute_base_latent(self, seed: int = 42) -> torch.Tensor:
        if self.G is None:
            self.load_model()
        z = torch.from_numpy(
            np.random.RandomState(seed).randn(1, self.G.z_dim)
        ).float().to(self.device)
        with torch.no_grad():
            self.w_base = self.G.mapping(z, None)
        del z
        _flush_mps()
        return self.w_base

    def generate_texture(
        self,
        rms_total: float,
        w_override: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Generate a single StyleGAN3 texture frame.
        Returns (H, W, 3) uint8 numpy array.
        """
        if self.G is None:
            self.load_model()

        w = w_override if w_override is not None else self.w_base
        if w is None:
            raise RuntimeError("Call compute_base_latent first.")

        w_gpu = w.float().to(self.device)

        with torch.no_grad():
            img = self.G.synthesis(w_gpu)

        result = ((img.clamp(-1, 1) + 1) * 127.5)[0]
        result = result.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        del img, w_gpu
        _flush_mps()

        # Resize from native resolution (e.g. 1024) to texture size (e.g. 256)
        if STYLEGAN3_TEXTURE_SIZE != STYLEGAN3_RESOLUTION:
            pil_img = PILImage.fromarray(result)
            pil_img = pil_img.resize(
                (STYLEGAN3_TEXTURE_SIZE, STYLEGAN3_TEXTURE_SIZE),
                PILImage.LANCZOS,
            )
            result = np.array(pil_img)

        return result

    def generate_w_sequence(
        self, base_w, num_frames, audio_rms, smoothing=0.5,
    ) -> list[torch.Tensor]:
        """
        W sequence with DRAMATIC variation — each frame must look visibly different.
        Uses cumulative random walk so the texture evolves over time,
        plus strong energy-based jumps on beats.
        """
        base = base_w.cpu().float()
        num_ws = base.shape[1]
        ws = []

        # Cumulative drift for smooth evolution
        drift = torch.zeros_like(base)

        for t in range(num_frames):
            energy = sum(
                r[min(t, len(r) - 1)] for r in audio_rms.values()
            ) / max(len(audio_rms), 1)

            # Smooth drift + energy-scaled jump
            drift += torch.randn_like(base) * 0.02  # slow evolution
            jump = torch.randn_like(base) * smoothing * energy  # beat reactivity

            # Extra punch on mid layers for dramatic color/shape changes
            mid_start = num_ws // 4
            mid_end = 3 * num_ws // 4
            jump[0, mid_start:mid_end] *= 2.5

            ws.append(base + drift + jump)
        return ws
