"""
FastAPI application entry point for the
7.1.4 Spatial Audio Visualization backend.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

import os
import sys

# MPS memory — MUST be set before torch is imported
# HIGH=1.0 uses all unified memory; LOW must be < HIGH
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.7")

# Add stylegan3 to path for dnnlib/legacy imports
stylegan3_dir = os.path.expanduser("~/stylegan3")
if stylegan3_dir not in sys.path:
    sys.path.insert(0, stylegan3_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.config import DEVICE

app = FastAPI(
    title="7.1.4 Spatial Audio Visualization API",
    description="StyleGAN3-based spatial audio-reactive video generation",
    version="1.0.0",
)

# CORS — allow the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "gpu_backend": "MPS" if "mps" in str(DEVICE) else str(DEVICE).upper(),
    }
