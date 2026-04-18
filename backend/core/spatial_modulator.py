"""
Spatial Latent Modulation Engine.

Combines 12-channel (x, y) marker coordinates with per-channel RMS energy
to create spatially-varying deformation fields that warp StyleGAN3's
intermediate feature maps. This is the core of the "audio-to-visual-motion"
pipeline described in the research concept.

Key idea:
  For each video frame t:
    1. Read RMS(t) for all 12 channels.
    2. For each channel, create a 2D Gaussian blob centered at (x_ch, y_ch)
       with amplitude proportional to RMS(t, ch).
    3. Sum all 12 blobs → Spatial Energy Map E(x, y, t).
    4. Convert E into a deformation (warp) field and an additive
       latent modulation signal that is injected into StyleGAN3's
       synthesis network.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class ChannelMarker:
    name: str
    x: float   # normalized [0, 1]
    y: float   # normalized [0, 1]


def build_gaussian_blob(
    cx: float,
    cy: float,
    amplitude: float,
    resolution: int,
    sigma: float = 0.08,
) -> torch.Tensor:
    """
    Create a 2D Gaussian blob on a (resolution x resolution) grid.
    cx, cy in [0, 1]. Returns shape (1, 1, H, W).
    """
    ys = torch.linspace(0, 1, resolution)
    xs = torch.linspace(0, 1, resolution)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    blob = amplitude * torch.exp(-dist_sq / (2 * sigma ** 2))
    return blob.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def build_spatial_energy_map(
    markers: list[ChannelMarker],
    rms_values: dict[str, float],
    resolution: int = 256,
    sigma: float = 0.08,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Sum 12 Gaussian blobs → single spatial energy map.
    Returns shape (1, 1, resolution, resolution), values in ~[0, 1].
    """
    energy = torch.zeros(1, 1, resolution, resolution, device=device, dtype=torch.float16)
    for m in markers:
        amp = rms_values.get(m.name, 0.0)
        if amp < 1e-6:
            continue
        blob = build_gaussian_blob(m.x, m.y, amp, resolution, sigma)
        energy += blob.to(device).half()
    energy = energy.clamp(0, 1)
    return energy


def energy_to_warp_field(
    energy_map: torch.Tensor,
    warp_strength: float = 0.15,
) -> torch.Tensor:
    """
    Convert a spatial energy map into a 2D displacement (warp) field.
    Uses gradient of the energy map as the deformation direction,
    scaled by energy magnitude.

    Returns a flow field of shape (1, H, W, 2) compatible with
    F.grid_sample's grid format.
    """
    _, _, H, W = energy_map.shape

    # Compute spatial gradients (Sobel-like)
    # dx
    dtype = energy_map.dtype
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype,
                       device=energy_map.device).reshape(1, 1, 3, 3)
    # dy
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype,
                       device=energy_map.device).reshape(1, 1, 3, 3)

    padded = F.pad(energy_map, (1, 1, 1, 1), mode="replicate")
    dx = F.conv2d(padded, kx)  # (1, 1, H, W)
    dy = F.conv2d(padded, ky)

    # Scale by energy magnitude and warp_strength
    dx = dx * energy_map * warp_strength
    dy = dy * energy_map * warp_strength

    # Build identity grid
    theta = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=dtype,
                         device=energy_map.device).unsqueeze(0)
    grid = F.affine_grid(theta, (1, 1, H, W), align_corners=True).to(dtype)  # (1, H, W, 2)

    # Add displacement
    flow = torch.stack([dx.squeeze(), dy.squeeze()], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    warped_grid = grid + flow

    return warped_grid


def modulate_latent_features(
    feature_maps: torch.Tensor,
    energy_map: torch.Tensor,
    warp_strength: float = 0.15,
    modulation_scale: float = 0.3,
) -> torch.Tensor:
    """
    Apply spatial modulation to StyleGAN3 intermediate feature maps.

    1. Warp: Deform the feature map using the energy-derived flow field.
    2. Additive modulation: Boost activations in high-energy regions.

    Args:
        feature_maps: (B, C, H, W) intermediate features from StyleGAN3.
        energy_map: (1, 1, He, We) spatial energy map.
        warp_strength: magnitude of geometric deformation.
        modulation_scale: magnitude of additive activation boost.

    Returns:
        Modulated feature maps, same shape as input.
    """
    B, C, H, W = feature_maps.shape

    # Resize energy map to match feature map spatial dims
    energy_resized = F.interpolate(
        energy_map, size=(H, W), mode="bilinear", align_corners=True
    )

    # 1. Warp
    warp_grid = energy_to_warp_field(energy_resized, warp_strength)
    # grid_sample expects (B, C, H, W) input and (B, H, W, 2) grid
    warp_grid_expanded = warp_grid.expand(B, -1, -1, -1)
    warped = F.grid_sample(
        feature_maps, warp_grid_expanded,
        mode="bilinear", padding_mode="border", align_corners=True
    )

    # 2. Additive modulation: scale activations by energy
    modulation = energy_resized.expand(B, C, -1, -1) * modulation_scale
    modulated = warped + warped * modulation

    return modulated
