"""
FastAPI routes for the Spatial Audio Visualization backend.

Endpoints:
  POST /api/analyze   — Extract per-channel RMS & onset from uploaded audio
  POST /api/bake      — Start video generation (returns SSE stream)
  GET  /api/video/{name} — Serve generated video files
"""

import json
import asyncio
import uuid
from pathlib import Path

from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from backend.config import UPLOAD_DIR, OUTPUT_DIR
from backend.core.audio_analyzer import analyze_audio
from backend.core.video_baker import VideoBaker
from backend.core.video_baker_physics import PhysicsVideoBaker

router = APIRouter(prefix="/api")


async def _save_upload(file: UploadFile, suffix: str = "") -> Path:
    """Save an uploaded file and return its path."""
    ext = Path(file.filename).suffix if file.filename else suffix
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    content = await file.read()
    dest.write_bytes(content)
    return dest


# ──────────────────────────────────────────────
# POST /api/analyze
# ──────────────────────────────────────────────

@router.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    """
    Upload a 7.1.4 multi-channel audio file.
    Returns per-channel RMS and onset envelope data as JSON.
    """
    audio_ext = Path(audio.filename).suffix if audio.filename else ".wav"
    audio_path = await _save_upload(audio, audio_ext)
    try:
        result = analyze_audio(audio_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ──────────────────────────────────────────────
# POST /api/bake  (SSE stream)
# ──────────────────────────────────────────────

@router.post("/bake")
async def bake(
    audio: UploadFile = File(...),
    channel_positions: str = Form(...),
    image: Optional[UploadFile] = File(None),
):
    """
    Start the video bake process.
    `channel_positions` is a JSON string: {"L": {"x": 0.25, "y": 0.45}, ...}

    Returns a Server-Sent Events (SSE) stream with progress updates:
      data: {"status": "processing", "message": "...", "progress": 0.42}
      data: {"status": "complete", "video_url": "/api/video/output_xxx.mp4"}
      data: {"status": "error", "message": "..."}
    """
    # Parse channel positions
    try:
        positions = json.loads(channel_positions)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid channel_positions JSON")

    # Save uploads
    image_path = None
    if image is not None:
        image_path = await _save_upload(image, ".png")
    audio_ext = Path(audio.filename).suffix if audio.filename else ".wav"
    audio_path = await _save_upload(audio, audio_ext)

    output_name = f"output_{uuid.uuid4().hex[:8]}"

    async def event_stream():
        baker = VideoBaker()
        progress_queue = asyncio.Queue()

        async def progress_callback(message: str, progress: float):
            await progress_queue.put({
                "status": "processing",
                "message": message,
                "progress": round(progress, 4),
            })

        async def run_bake():
            try:
                video_path = await baker.bake(
                    image_path=image_path,
                    audio_path=audio_path,
                    channel_positions=positions,
                    output_name=output_name,
                    progress_callback=progress_callback,
                )
                await progress_queue.put({
                    "status": "complete",
                    "video_url": f"/api/video/{video_path.name}",
                    "progress": 1.0,
                })
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                await progress_queue.put({
                    "status": "error",
                    "message": f"{e}\n{tb}",
                    "progress": -1,
                })
            finally:
                await progress_queue.put(None)  # sentinel

        task = asyncio.create_task(run_bake())

        while True:
            item = await progress_queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

        await task  # ensure task completes

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ──────────────────────────────────────────────
# POST /api/bake_physics  (SSE stream, no StyleGAN)
# ──────────────────────────────────────────────

@router.post("/bake_physics")
async def bake_physics(
    audio: UploadFile = File(...),
    channel_positions: str = Form(...),
    image: Optional[UploadFile] = File(None),
):
    """
    Physics-based media art bake (fluid particle field).
    Same request/response shape as /api/bake; swaps the StyleGAN
    engine for a CPU particle simulator driven by per-channel RMS.
    """
    try:
        positions = json.loads(channel_positions)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid channel_positions JSON")

    image_path = None
    if image is not None:
        image_path = await _save_upload(image, ".png")
    audio_ext = Path(audio.filename).suffix if audio.filename else ".wav"
    audio_path = await _save_upload(audio, audio_ext)

    output_name = f"physics_{uuid.uuid4().hex[:8]}"

    async def event_stream():
        baker = PhysicsVideoBaker()
        progress_queue = asyncio.Queue()

        async def progress_callback(message: str, progress: float):
            await progress_queue.put({
                "status": "processing",
                "message": message,
                "progress": round(progress, 4),
            })

        async def run_bake():
            try:
                video_path = await baker.bake(
                    image_path=image_path,
                    audio_path=audio_path,
                    channel_positions=positions,
                    output_name=output_name,
                    progress_callback=progress_callback,
                )
                await progress_queue.put({
                    "status": "complete",
                    "video_url": f"/api/video/{video_path.name}",
                    "progress": 1.0,
                })
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                await progress_queue.put({
                    "status": "error",
                    "message": f"{e}\n{tb}",
                    "progress": -1,
                })
            finally:
                await progress_queue.put(None)

        task = asyncio.create_task(run_bake())

        while True:
            item = await progress_queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

        await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ──────────────────────────────────────────────
# GET /api/video/{filename}
# ──────────────────────────────────────────────

@router.get("/video/{filename}")
async def get_video(filename: str):
    """Serve a generated video file."""
    video_path = OUTPUT_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")
