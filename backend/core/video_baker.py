"""
Video Bake Pipeline — 5-Screen Spatial Audio Visualization.

Produces a video with the layout:  [L2] [L1] [CENTER] [R1] [R2]

Each frame:
  1. Composite 5-screen layout from the user's reference image
     - CENTER: full image
     - L1/L2: left 50% crop
     - R1/R2: right 50% crop
  2. Per-channel audio-reactive effects at marker positions:
     - Radial warp (displacement) scaled by RMS
     - Brightness pulse / glow
     - StyleGAN3 texture blended at high-energy spots
  3. Channel label overlay + energy ring indicators
"""

import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import map_coordinates

from backend.config import (
    DEVICE, VIDEO_FPS, VIDEO_CODEC, VIDEO_CRF,
    AUDIO_SR, AUDIO_HOP_LENGTH, CHANNEL_NAMES, OUTPUT_DIR,
)
from backend.core.audio_analyzer import load_multichannel, extract_rms
from backend.core.spatial_modulator import ChannelMarker
from backend.core.stylegan3_engine import StyleGAN3Engine

# Output resolution
OUT_W, OUT_H = 1920, 1080
# Screen proportions: L2(7.5%) L1(7.5%) CENTER(70%) R1(7.5%) R2(7.5%)
SCREEN_RATIOS = [0.075, 0.075, 0.70, 0.075, 0.075]
SCREEN_NAMES = ["L2", "L1", "CENTER", "R1", "R2"]
GAP = 4  # pixels between screens


class VideoBaker:

    def __init__(self):
        self.engine = StyleGAN3Engine(device=DEVICE)

    async def bake(
        self,
        image_path: str | Path,
        audio_path: str | Path,
        channel_positions: dict[str, dict],
        output_name: str = "output",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Path:
        output_video = OUTPUT_DIR / f"{output_name}.mp4"

        async def report(msg: str, p: float):
            if progress_callback:
                await progress_callback(msg, p)

        # --- Step 1: Load model + reference image ---
        await report("Loading StyleGAN3 model...", 0.0)
        await asyncio.to_thread(self.engine.load_model)

        await report("Computing base latent vector...", 0.03)
        w_base = await asyncio.to_thread(self.engine.compute_base_latent, 42)

        await report("Preparing 5-screen layout...", 0.05)
        if image_path is not None:
            ref_img = Image.open(str(image_path)).convert("RGB")
        else:
            ref_img = Image.new("RGB", (OUT_W, OUT_H), (0, 0, 0))
        screen_images = self._build_screen_images(ref_img)

        # --- Step 2: Analyze audio ---
        await report("Analyzing 12-channel audio (RMS extraction)...", 0.10)
        audio = await asyncio.to_thread(load_multichannel, audio_path)
        rms_data = await asyncio.to_thread(extract_rms, audio)
        num_samples = audio.shape[1]
        duration = num_samples / AUDIO_SR
        num_video_frames = int(duration * VIDEO_FPS)

        markers = []
        for ch_name in CHANNEL_NAMES:
            pos = channel_positions.get(ch_name, {"x": 0.5, "y": 0.5})
            markers.append(ChannelMarker(name=ch_name, x=pos["x"], y=pos["y"]))

        # --- Step 3: Generate W sequence for StyleGAN textures ---
        await report("Computing audio-reactive latent sequence...", 0.15)
        rms_lists = {name: rms_data[name].tolist() for name in CHANNEL_NAMES}
        w_sequence = await asyncio.to_thread(
            self.engine.generate_w_sequence, w_base, num_video_frames, rms_lists, 0.3
        )

        # --- Step 4: Generate frames ---
        status_messages = [
            "Generating spatial weight maps...",
            "Applying audio-reactive warping...",
            "Blending StyleGAN3 textures...",
            "Compositing 5-screen layout...",
            "Rendering frames (MPS accelerated)...",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)

            for frame_idx in range(num_video_frames):
                audio_frame_idx = int(
                    frame_idx * (AUDIO_SR / VIDEO_FPS) / AUDIO_HOP_LENGTH
                )

                frame_rms = {}
                for ch_name in CHANNEL_NAMES:
                    rms_arr = rms_data[ch_name]
                    idx = min(audio_frame_idx, len(rms_arr) - 1)
                    frame_rms[ch_name] = float(rms_arr[idx])

                total_energy = sum(frame_rms.values()) / max(len(frame_rms), 1)

                # Generate StyleGAN texture for this frame
                stylegan_texture = await asyncio.to_thread(
                    self.engine.generate_texture, total_energy, w_sequence[frame_idx],
                )

                # Composite the full frame
                frame = await asyncio.to_thread(
                    self._composite_frame,
                    screen_images, markers, frame_rms,
                    stylegan_texture, total_energy,
                )

                frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
                await asyncio.to_thread(
                    Image.fromarray(frame).save, frame_path
                )
                del frame, stylegan_texture

                if frame_idx % 5 == 0:
                    p = 0.20 + 0.65 * (frame_idx / num_video_frames)
                    msg = status_messages[frame_idx % len(status_messages)]
                    await report(msg, p)

            # --- Step 5: Encode ---
            await report("Encoding with FFmpeg (H.264)...", 0.88)
            await asyncio.to_thread(
                self._encode_video, frames_dir, audio_path, output_video
            )

        await report("Finalizing output...", 0.95)
        return output_video

    def _build_screen_images(self, ref_img: Image.Image) -> list[np.ndarray]:
        """
        Build 5 screen images from reference:
          L2, L1: left 50% crop, stretched to screen size
          CENTER: full image
          R1, R2: right 50% crop, stretched to screen size
        Returns list of 5 numpy arrays (H, W, 3) float32 [0,1].
        """
        w_total = OUT_W - GAP * 4
        screens = []

        for i, ratio in enumerate(SCREEN_RATIOS):
            sw = int(w_total * ratio)
            sh = OUT_H

            if SCREEN_NAMES[i] == "CENTER":
                crop = ref_img.copy()
            elif SCREEN_NAMES[i] in ("L1", "L2"):
                # Left 50%
                iw, ih = ref_img.size
                crop = ref_img.crop((0, 0, iw // 2, ih))
            else:
                # Right 50%
                iw, ih = ref_img.size
                crop = ref_img.crop((iw // 2, 0, iw, ih))

            crop = crop.resize((sw, sh), Image.LANCZOS)
            screens.append(np.array(crop).astype(np.float32) / 255.0)

        return screens

    def _composite_frame(
        self,
        screen_images: list[np.ndarray],
        markers: list[ChannelMarker],
        rms_values: dict[str, float],
        stylegan_texture: np.ndarray,
        total_energy: float,
    ) -> np.ndarray:
        """
        Composite one video frame:
          1. Place 5 screens side by side on black background
          2. Apply per-channel warp + glow effects
          3. Blend StyleGAN texture at high-energy channel positions
          4. Draw channel markers with energy rings
        """
        canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.float32)

        # --- Place screens ---
        w_total = OUT_W - GAP * 4
        x_offset = 0
        screen_rects = []  # (x, y, w, h) for each screen

        for i, ratio in enumerate(SCREEN_RATIOS):
            sw = int(w_total * ratio)
            sh = OUT_H
            screen_rects.append((x_offset, 0, sw, sh))
            canvas[0:sh, x_offset:x_offset+sw] = screen_images[i]
            x_offset += sw + GAP

        # --- Apply audio-reactive effects ---
        # Warp the canvas based on channel positions
        canvas = self._apply_spatial_warp(canvas, markers, rms_values)

        # Brightness pulse at channel positions
        canvas = self._apply_energy_glow(canvas, markers, rms_values)

        # Blend StyleGAN texture at high-energy spots
        canvas = self._blend_stylegan_texture(
            canvas, stylegan_texture, markers, rms_values
        )

        # --- Draw screen borders ---
        x_offset = 0
        for i, ratio in enumerate(SCREEN_RATIOS):
            sw = int(w_total * ratio)
            # Thin border
            canvas[0:2, x_offset:x_offset+sw] = [0.2, 0.4, 0.8] if SCREEN_NAMES[i] == "CENTER" else [0.3, 0.3, 0.3]
            canvas[OUT_H-2:OUT_H, x_offset:x_offset+sw] = [0.2, 0.4, 0.8] if SCREEN_NAMES[i] == "CENTER" else [0.3, 0.3, 0.3]
            canvas[0:OUT_H, x_offset:x_offset+2] = [0.3, 0.3, 0.3]
            canvas[0:OUT_H, x_offset+sw-2:x_offset+sw] = [0.3, 0.3, 0.3]
            x_offset += sw + GAP

        # --- Draw channel markers ---
        canvas_uint8 = np.clip(canvas * 255, 0, 255).astype(np.uint8)
        pil_frame = Image.fromarray(canvas_uint8)
        draw = ImageDraw.Draw(pil_frame)

        for m in markers:
            amp = rms_values.get(m.name, 0.0)
            px = int(m.x * OUT_W)
            py = int(m.y * OUT_H)

            # Energy ring
            ring_r = int(12 + amp * 25)
            ring_color = self._channel_color(m.name, amp)
            draw.ellipse(
                [px - ring_r, py - ring_r, px + ring_r, py + ring_r],
                outline=ring_color, width=2
            )

            # Label
            draw.text(
                (px - 8, py - 6), m.name,
                fill=(255, 255, 255), font=None
            )

        return np.array(pil_frame)

    def _apply_spatial_warp(
        self, canvas, markers, rms_values
    ) -> np.ndarray:
        """Radial displacement at each channel position, scaled by RMS."""
        H, W = canvas.shape[:2]
        dy_field = np.zeros((H, W), dtype=np.float32)
        dx_field = np.zeros((H, W), dtype=np.float32)

        ys = np.linspace(0, 1, H)
        xs = np.linspace(0, 1, W)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')

        for m in markers:
            amp = rms_values.get(m.name, 0.0)
            if amp < 0.05:
                continue

            sigma = 0.08
            dist_sq = (xx - m.x) ** 2 + (yy - m.y) ** 2
            gaussian = np.exp(-dist_sq / (2 * sigma ** 2))

            dir_x = xx - m.x
            dir_y = yy - m.y
            mag = np.sqrt(dir_x ** 2 + dir_y ** 2) + 1e-8

            strength = amp * 30  # pixels of displacement
            dx_field += (dir_x / mag) * gaussian * strength
            dy_field += (dir_y / mag) * gaussian * strength

        row_coords = np.clip(np.arange(H)[:, None] + dy_field, 0, H - 1)
        col_coords = np.clip(np.arange(W)[None, :] + dx_field, 0, W - 1)

        warped = np.zeros_like(canvas)
        for c in range(3):
            warped[:, :, c] = map_coordinates(
                canvas[:, :, c], [row_coords, col_coords],
                order=1, mode='reflect'
            )
        return warped

    def _apply_energy_glow(
        self, canvas, markers, rms_values
    ) -> np.ndarray:
        """Brightness boost at active channel positions."""
        H, W = canvas.shape[:2]
        ys = np.linspace(0, 1, H)
        xs = np.linspace(0, 1, W)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')

        for m in markers:
            amp = rms_values.get(m.name, 0.0)
            if amp < 0.1:
                continue

            sigma = 0.06
            dist_sq = (xx - m.x) ** 2 + (yy - m.y) ** 2
            glow = np.exp(-dist_sq / (2 * sigma ** 2)) * amp * 0.6

            r, g, b = self._channel_color_float(m.name)
            canvas[:, :, 0] += glow * r
            canvas[:, :, 1] += glow * g
            canvas[:, :, 2] += glow * b

        return np.clip(canvas, 0, 1)

    def _blend_stylegan_texture(
        self, canvas, texture, markers, rms_values
    ) -> np.ndarray:
        """
        Blend StyleGAN3-generated texture onto the canvas.

        - Base layer: 15% texture everywhere (subtle generative atmosphere)
        - Per-channel blobs: up to 80% texture at high-energy marker positions
        - Color tinting: each channel's blob is tinted with its color
        """
        H, W = canvas.shape[:2]
        tex = np.array(
            Image.fromarray(texture).resize((W, H), Image.LANCZOS)
        ).astype(np.float32) / 255.0

        ys = np.linspace(0, 1, H)
        xs = np.linspace(0, 1, W)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')

        # 1. Base atmospheric layer — always visible
        total_energy = sum(rms_values.values()) / max(len(rms_values), 1)
        base_alpha = 0.08 + total_energy * 0.12  # 8~20% base blend
        blended = canvas * (1 - base_alpha) + tex * base_alpha

        # 2. Per-channel strong blending with color tint
        for m in markers:
            amp = rms_values.get(m.name, 0.0)
            if amp < 0.02:
                continue

            sigma = 0.10  # wider spread
            dist_sq = (xx - m.x) ** 2 + (yy - m.y) ** 2
            blob = np.exp(-dist_sq / (2 * sigma ** 2))

            # Blend strength: scales strongly with amplitude
            strength = amp * 0.8  # up to 80% texture at full amplitude
            mask = (blob * strength)[:, :, np.newaxis]

            # Tint the texture with channel color
            r, g, b = self._channel_color_float(m.name)
            tinted_tex = tex.copy()
            tinted_tex[:, :, 0] = tex[:, :, 0] * 0.4 + tex[:, :, 0] * r * 0.6
            tinted_tex[:, :, 1] = tex[:, :, 1] * 0.4 + tex[:, :, 1] * g * 0.6
            tinted_tex[:, :, 2] = tex[:, :, 2] * 0.4 + tex[:, :, 2] * b * 0.6

            blended = blended * (1 - mask) + tinted_tex * mask

        return np.clip(blended, 0, 1)

    def _channel_color(self, name: str, amp: float) -> tuple:
        """Get channel color as RGB tuple (0-255), brighter with amplitude."""
        colors = {
            'L': (59, 130, 246), 'R': (239, 68, 68), 'C': (34, 197, 94),
            'LFE': (245, 158, 11), 'Ls': (99, 102, 241), 'Rs': (236, 72, 153),
            'Lrs': (139, 92, 246), 'Rrs': (244, 63, 94),
            'Ltf': (6, 182, 212), 'Rtf': (249, 115, 22),
            'Ltr': (20, 184, 166), 'Rtr': (232, 121, 249),
        }
        base = colors.get(name, (150, 150, 150))
        boost = min(amp * 1.5, 1.0)
        return tuple(min(255, int(c * (0.5 + boost * 0.5))) for c in base)

    def _channel_color_float(self, name: str) -> tuple:
        """Channel color as float (0-1)."""
        c = self._channel_color(name, 1.0)
        return (c[0] / 255, c[1] / 255, c[2] / 255)

    def _encode_video(self, frames_dir, audio_path, output_path):
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(VIDEO_FPS),
            "-i", str(frames_dir / "frame_%06d.png"),
            "-i", str(audio_path),
            "-c:v", VIDEO_CODEC,
            "-crf", str(VIDEO_CRF),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "256k",
            "-ac", "2", "-shortest",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
