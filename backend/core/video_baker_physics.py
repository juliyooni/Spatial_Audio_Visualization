"""
Physics-based Media Art Video Baker (no StyleGAN).

Style: Fluid Particle Field
  - Per-channel particle emitters at user-placed marker positions.
  - Per-channel RMS drives: emission rate, initial velocity, glow size, color intensity.
  - Particles obey simple 2D physics:
      * initial radial velocity from the emitter (with jitter)
      * drag (exponential velocity decay)
      * gravity-like drift toward canvas center (very weak)
      * life-based alpha fade
  - Global bloom-style additive blending over a blurred, dimmed reference image.
  - Screen layout unchanged: [L2][L1][CENTER][R1][R2]
"""

import asyncio
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from backend.config import (
    VIDEO_FPS, VIDEO_CODEC, VIDEO_CRF,
    AUDIO_SR, AUDIO_HOP_LENGTH, CHANNEL_NAMES, OUTPUT_DIR,
)
from backend.core.audio_analyzer import load_multichannel, extract_rms

# Output resolution (same as StyleGAN pipeline)
OUT_W, OUT_H = 1920, 1080
# Screen proportions (unchanged): L2(7.5%) L1(7.5%) CENTER(70%) R1(7.5%) R2(7.5%)
SCREEN_RATIOS = [0.075, 0.075, 0.70, 0.075, 0.075]
SCREEN_NAMES = ["L2", "L1", "CENTER", "R1", "R2"]
GAP = 4

# Physics tunables
MAX_PARTICLES = 6000
EMIT_SCALE = 35           # max new particles / channel / frame at RMS=1
BASE_SPEED = 260          # px/s initial speed when RMS=1
SPEED_JITTER = 0.6        # fraction of speed randomized
DRAG = 2.2                # per-second velocity decay factor (v *= exp(-DRAG*dt))
CENTER_PULL = 8.0         # px/s^2 toward screen center
LIFE_MIN, LIFE_MAX = 0.8, 2.2  # seconds
PARTICLE_R_MIN, PARTICLE_R_MAX = 1.5, 4.5  # pixel radius


CHANNEL_COLORS = {
    'L':   (59, 130, 246), 'R':   (239, 68, 68), 'C':   (34, 197, 94),
    'LFE': (245, 158, 11), 'Ls':  (99, 102, 241), 'Rs': (236, 72, 153),
    'Lrs': (139, 92, 246), 'Rrs': (244, 63, 94),
    'Ltf': (6, 182, 212),  'Rtf': (249, 115, 22),
    'Ltr': (20, 184, 166), 'Rtr': (232, 121, 249),
}


@dataclass
class ParticleSystem:
    """Struct-of-arrays particle pool for speed."""
    cap: int = MAX_PARTICLES
    n: int = 0
    x: np.ndarray = field(default_factory=lambda: np.zeros(MAX_PARTICLES, dtype=np.float32))
    y: np.ndarray = field(default_factory=lambda: np.zeros(MAX_PARTICLES, dtype=np.float32))
    vx: np.ndarray = field(default_factory=lambda: np.zeros(MAX_PARTICLES, dtype=np.float32))
    vy: np.ndarray = field(default_factory=lambda: np.zeros(MAX_PARTICLES, dtype=np.float32))
    life: np.ndarray = field(default_factory=lambda: np.zeros(MAX_PARTICLES, dtype=np.float32))
    max_life: np.ndarray = field(default_factory=lambda: np.ones(MAX_PARTICLES, dtype=np.float32))
    radius: np.ndarray = field(default_factory=lambda: np.ones(MAX_PARTICLES, dtype=np.float32))
    r: np.ndarray = field(default_factory=lambda: np.zeros(MAX_PARTICLES, dtype=np.float32))
    g: np.ndarray = field(default_factory=lambda: np.zeros(MAX_PARTICLES, dtype=np.float32))
    b: np.ndarray = field(default_factory=lambda: np.zeros(MAX_PARTICLES, dtype=np.float32))

    def emit(self, count: int, cx: float, cy: float, speed: float,
             color: tuple, rng: np.random.Generator):
        if count <= 0:
            return
        # Drop oldest if overfull
        free = self.cap - self.n
        if count > free:
            # Compact: remove first `count - free` particles (oldest end of queue)
            drop = count - free
            self._drop_front(drop)
        i0, i1 = self.n, self.n + count
        self.x[i0:i1] = cx
        self.y[i0:i1] = cy
        # Random directions, random speed magnitudes
        angles = rng.uniform(0, 2 * np.pi, count).astype(np.float32)
        mags = speed * (1.0 - SPEED_JITTER + rng.random(count).astype(np.float32) * SPEED_JITTER * 2)
        self.vx[i0:i1] = np.cos(angles) * mags
        self.vy[i0:i1] = np.sin(angles) * mags
        ml = rng.uniform(LIFE_MIN, LIFE_MAX, count).astype(np.float32)
        self.max_life[i0:i1] = ml
        self.life[i0:i1] = ml
        self.radius[i0:i1] = rng.uniform(PARTICLE_R_MIN, PARTICLE_R_MAX, count).astype(np.float32)
        self.r[i0:i1] = color[0] / 255.0
        self.g[i0:i1] = color[1] / 255.0
        self.b[i0:i1] = color[2] / 255.0
        self.n = i1

    def _drop_front(self, k: int):
        k = min(k, self.n)
        if k <= 0:
            return
        # Shift arrays left by k
        for arr in (self.x, self.y, self.vx, self.vy, self.life,
                    self.max_life, self.radius, self.r, self.g, self.b):
            arr[: self.n - k] = arr[k: self.n]
        self.n -= k

    def step(self, dt: float):
        if self.n == 0:
            return
        n = self.n
        # Center pull (weak gravity toward canvas center)
        self.vx[:n] += (OUT_W * 0.5 - self.x[:n]) * (CENTER_PULL / max(OUT_W, 1)) * dt
        self.vy[:n] += (OUT_H * 0.5 - self.y[:n]) * (CENTER_PULL / max(OUT_H, 1)) * dt
        # Drag
        decay = np.exp(-DRAG * dt, dtype=np.float32)
        self.vx[:n] *= decay
        self.vy[:n] *= decay
        # Integrate
        self.x[:n] += self.vx[:n] * dt
        self.y[:n] += self.vy[:n] * dt
        # Age
        self.life[:n] -= dt
        # Cull dead / offscreen
        alive = (
            (self.life[:n] > 0)
            & (self.x[:n] > -20) & (self.x[:n] < OUT_W + 20)
            & (self.y[:n] > -20) & (self.y[:n] < OUT_H + 20)
        )
        if not alive.all():
            idx = np.nonzero(alive)[0]
            m = idx.size
            for arr in (self.x, self.y, self.vx, self.vy, self.life,
                        self.max_life, self.radius, self.r, self.g, self.b):
                arr[:m] = arr[:n][idx]
            self.n = m


def _marker_pixel(pos: dict) -> tuple[int, int]:
    return int(pos.get("x", 0.5) * OUT_W), int(pos.get("y", 0.5) * OUT_H)


def _build_screen_images(ref_img: Image.Image) -> list[np.ndarray]:
    """Same 5-screen split as the StyleGAN baker."""
    w_total = OUT_W - GAP * 4
    screens = []
    for i, ratio in enumerate(SCREEN_RATIOS):
        sw = int(w_total * ratio)
        sh = OUT_H
        if SCREEN_NAMES[i] == "CENTER":
            crop = ref_img.copy()
        elif SCREEN_NAMES[i] in ("L1", "L2"):
            iw, ih = ref_img.size
            crop = ref_img.crop((0, 0, iw // 2, ih))
        else:
            iw, ih = ref_img.size
            crop = ref_img.crop((iw // 2, 0, iw, ih))
        crop = crop.resize((sw, sh), Image.LANCZOS)
        screens.append(np.array(crop).astype(np.float32) / 255.0)
    return screens


def _build_background(screen_images: list[np.ndarray]) -> np.ndarray:
    """Blurred, dimmed reference as the art background; keeps screen borders."""
    canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.float32)
    w_total = OUT_W - GAP * 4
    x_offset = 0
    for i, ratio in enumerate(SCREEN_RATIOS):
        sw = int(w_total * ratio)
        canvas[0:OUT_H, x_offset:x_offset + sw] = screen_images[i]
        x_offset += sw + GAP
    # Blur + dim for an ambient base
    pil = Image.fromarray(np.clip(canvas * 255, 0, 255).astype(np.uint8))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=18))
    bg = np.array(pil).astype(np.float32) / 255.0
    bg *= 0.28  # dim
    return bg


def _draw_screen_borders(canvas: np.ndarray):
    w_total = OUT_W - GAP * 4
    x_offset = 0
    for i, ratio in enumerate(SCREEN_RATIOS):
        sw = int(w_total * ratio)
        edge = [0.25, 0.45, 0.85] if SCREEN_NAMES[i] == "CENTER" else [0.3, 0.3, 0.35]
        canvas[0:2, x_offset:x_offset + sw] = edge
        canvas[OUT_H - 2:OUT_H, x_offset:x_offset + sw] = edge
        canvas[0:OUT_H, x_offset:x_offset + 2] = edge
        canvas[0:OUT_H, x_offset + sw - 2:x_offset + sw] = edge
        x_offset += sw + GAP


def _splat_particles(canvas: np.ndarray, ps: ParticleSystem):
    """Additive gaussian-ish splats for each particle (vectorized stamping)."""
    if ps.n == 0:
        return
    n = ps.n
    # Alpha from remaining-life
    alpha = (ps.life[:n] / np.maximum(ps.max_life[:n], 1e-6)).astype(np.float32)
    alpha = np.clip(alpha, 0.0, 1.0)

    xs = ps.x[:n].astype(np.int32)
    ys = ps.y[:n].astype(np.int32)
    rs = ps.radius[:n]

    H, W, _ = canvas.shape
    # Use a small fixed set of stamp kernels by radius bucket for speed
    # Bucket radii into a handful of sizes (2,3,4,5)
    rad_int = np.clip(np.round(rs).astype(np.int32), 2, 5)
    # Precompute kernels
    kernels = {}
    for k in (2, 3, 4, 5):
        size = 2 * k + 1
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        d2 = (xx ** 2 + yy ** 2).astype(np.float32)
        kernels[k] = np.exp(-d2 / (2 * (k * 0.6) ** 2)).astype(np.float32)

    for i in range(n):
        x, y = xs[i], ys[i]
        if x < 0 or y < 0 or x >= W or y >= H:
            continue
        k = int(rad_int[i])
        stamp = kernels[k]
        size = 2 * k + 1
        x0, y0 = x - k, y - k
        x1, y1 = x0 + size, y0 + size
        sx0, sy0 = max(0, -x0), max(0, -y0)
        sx1, sy1 = size - max(0, x1 - W), size - max(0, y1 - H)
        dx0, dy0 = max(x0, 0), max(y0, 0)
        dx1, dy1 = min(x1, W), min(y1, H)
        if dx1 <= dx0 or dy1 <= dy0:
            continue
        sub = stamp[sy0:sy1, sx0:sx1] * alpha[i]
        canvas[dy0:dy1, dx0:dx1, 0] += sub * ps.r[i]
        canvas[dy0:dy1, dx0:dx1, 1] += sub * ps.g[i]
        canvas[dy0:dy1, dx0:dx1, 2] += sub * ps.b[i]


def _apply_bloom(canvas: np.ndarray) -> np.ndarray:
    """Cheap bloom: threshold bright regions, blur, add back."""
    pil = Image.fromarray(np.clip(canvas * 255, 0, 255).astype(np.uint8))
    bright = np.clip((canvas - 0.55) * 1.8, 0, 1)
    bright_pil = Image.fromarray((bright * 255).astype(np.uint8))
    bloom = bright_pil.filter(ImageFilter.GaussianBlur(radius=14))
    bloom_arr = np.array(bloom).astype(np.float32) / 255.0
    out = np.clip(np.array(pil).astype(np.float32) / 255.0 + bloom_arr * 0.9, 0, 1)
    return out


def _draw_markers(frame_uint8: np.ndarray, channel_positions: dict,
                  rms_values: dict) -> np.ndarray:
    pil = Image.fromarray(frame_uint8)
    draw = ImageDraw.Draw(pil)
    for ch_name in CHANNEL_NAMES:
        pos = channel_positions.get(ch_name)
        if pos is None:
            continue
        amp = rms_values.get(ch_name, 0.0)
        px, py = _marker_pixel(pos)
        ring_r = int(10 + amp * 22)
        color = CHANNEL_COLORS.get(ch_name, (200, 200, 200))
        boost = 0.5 + min(amp, 1.0) * 0.5
        ring_color = tuple(min(255, int(c * boost)) for c in color)
        draw.ellipse([px - ring_r, py - ring_r, px + ring_r, py + ring_r],
                     outline=ring_color, width=2)
        draw.text((px - 10, py - 7), ch_name, fill=(255, 255, 255))
    return np.array(pil)


class PhysicsVideoBaker:
    """StyleGAN-free, physics-driven media art baker."""

    async def bake(
        self,
        image_path: Optional[str | Path],
        audio_path: str | Path,
        channel_positions: dict[str, dict],
        output_name: str = "output",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Path:
        output_video = OUTPUT_DIR / f"{output_name}.mp4"

        async def report(msg: str, p: float):
            if progress_callback:
                await progress_callback(msg, p)

        # --- Step 1: Background art layer from reference ---
        await report("Preparing 5-screen layout...", 0.02)
        if image_path is not None:
            ref_img = Image.open(str(image_path)).convert("RGB")
        else:
            ref_img = Image.new("RGB", (OUT_W, OUT_H), (10, 10, 18))
        screen_images = _build_screen_images(ref_img)
        background = _build_background(screen_images)

        # --- Step 2: Audio analysis ---
        await report("Analyzing 12-channel audio (RMS extraction)...", 0.08)
        audio = await asyncio.to_thread(load_multichannel, audio_path)
        rms_data = await asyncio.to_thread(extract_rms, audio)
        num_samples = audio.shape[1]
        duration = num_samples / AUDIO_SR
        num_video_frames = int(duration * VIDEO_FPS)
        dt = 1.0 / VIDEO_FPS

        # --- Step 3: Physics simulation + frame rendering ---
        await report("Initializing particle system...", 0.12)
        ps = ParticleSystem()
        rng = np.random.default_rng(1337)

        status_messages = [
            "Simulating fluid particle field...",
            "Emitting per-channel bursts...",
            "Integrating velocities with drag...",
            "Compositing additive bloom...",
            "Rendering frames...",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)

            for frame_idx in range(num_video_frames):
                audio_frame_idx = int(
                    frame_idx * (AUDIO_SR / VIDEO_FPS) / AUDIO_HOP_LENGTH
                )
                frame_rms = {}
                for ch_name in CHANNEL_NAMES:
                    arr = rms_data[ch_name]
                    idx = min(audio_frame_idx, len(arr) - 1)
                    frame_rms[ch_name] = float(arr[idx])

                # Emit per channel proportional to RMS
                for ch_name in CHANNEL_NAMES:
                    pos = channel_positions.get(ch_name)
                    if pos is None:
                        continue
                    amp = frame_rms[ch_name]
                    if amp < 0.04:
                        continue
                    # Number of new particles this frame
                    count = int(EMIT_SCALE * amp * amp)  # quadratic feel
                    if count <= 0:
                        continue
                    cx, cy = _marker_pixel(pos)
                    speed = BASE_SPEED * (0.4 + amp)
                    color = CHANNEL_COLORS.get(ch_name, (200, 200, 200))
                    ps.emit(count, cx, cy, speed, color, rng)

                # Physics step
                ps.step(dt)

                # Render: start from background, splat particles, bloom
                canvas = background.copy()
                _splat_particles(canvas, ps)
                canvas = _apply_bloom(canvas)
                _draw_screen_borders(canvas)

                frame_uint8 = np.clip(canvas * 255, 0, 255).astype(np.uint8)
                frame_uint8 = _draw_markers(frame_uint8, channel_positions, frame_rms)

                frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
                await asyncio.to_thread(Image.fromarray(frame_uint8).save, frame_path)

                if frame_idx % 5 == 0:
                    p = 0.15 + 0.75 * (frame_idx / max(num_video_frames, 1))
                    msg = status_messages[frame_idx % len(status_messages)]
                    await report(msg, p)

            # --- Step 4: Encode ---
            await report("Encoding with FFmpeg (H.264)...", 0.92)
            await asyncio.to_thread(
                self._encode_video, frames_dir, audio_path, output_video
            )

        await report("Finalizing output...", 0.98)
        return output_video

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
