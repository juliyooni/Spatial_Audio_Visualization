"""K-pop Narrative & Visual Mapping pipeline.

`kpop_visual_mapping.ipynb`의 함수들을 모듈화한 것. iTunes 미리듣기 또는
YouTube에서 음원을 받아 Librosa 정량 feature를 뽑고, Genius 가사가 있으면
text 모드, 없으면 audio 모드로 Gemini를 호출해 비주얼 컨셉 JSON을 생성한다.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "music_analysis" / "kpop_visual_db.json"
TMP_AUDIO_DIR = REPO_ROOT / "samples" / "_tmp"

SONIC_TEXTURES = [
    "Synth-pop", "Brass-Heavy", "Glitch", "Orchestral",
    "Acoustic", "Rock-chic", "Hip-hop Beat", "Dreamy Pad",
]

NARRATIVE_ARCHETYPES = [
    "High-teen", "Cyberpunk", "Ethereal", "Narcissistic",
    "Mala-taste (Spicy)", "Retro-nostalgia", "Gothic-horror",
]

SYSTEM_PROMPT = f"""Role: K-pop Narrative & Visual Mapping Specialist

Task: 제공된 정량 데이터(Librosa)와 정성 데이터(Genius 가사/메타 또는 오디오)를 교차 분석하여,
K-pop 오디오 비주얼라이제이션용 표준 JSON을 생성하라.

Classification Standard (Consistency):
- Sonic Texture: {SONIC_TEXTURES} 중 1개 선택.
- Narrative Archetype: {NARRATIVE_ARCHETYPES} 중 1개 선택.
- Visual Symbol: 텍스트/오디오에서 도출된 시각적 오브젝트 3가지 (동물, 사물, 자연물 등).
- Color Mood: 메인 컬러(Hex 1개), 서브 컬러(Hex 2개).
  Librosa의 valence/energy를 색상의 명도/채도에 반영할 것.

Output Format (Strict JSON, no markdown, no commentary):
{{
  "sonic_texture": "",
  "narrative_archetype": "",
  "visual_symbols": ["", "", ""],
  "color_palette": {{"main": "#", "sub": ["#", "#"]}},
  "reasoning_brief": "분류 이유 요약 (한글, 2-3문장)"
}}

Constraint: 오직 JSON만 출력하고 다른 부연 설명은 하지 않는다."""


# ──────────────────────────────────────────────────────────────
# 환경 변수 / API 키
# ──────────────────────────────────────────────────────────────

def load_env():
    """`.env`를 읽고 (GENIUS_ACCESS_TOKEN, GEMINI_API_KEY) 반환."""
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    return os.getenv("GENIUS_ACCESS_TOKEN"), os.getenv("GEMINI_API_KEY")


# ──────────────────────────────────────────────────────────────
# 캐시
# ──────────────────────────────────────────────────────────────

def _normalize_key(title: str, artist: str) -> str:
    return f"{title.strip().lower()}__{artist.strip().lower()}"


def _load_db() -> dict:
    if not DB_PATH.exists():
        return {}
    try:
        return json.loads(DB_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_db(db: dict) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")


def get_from_cache(title: str, artist: str) -> Optional[dict]:
    return _load_db().get(_normalize_key(title, artist))


def save_to_cache(entry: dict) -> None:
    db = _load_db()
    db[_normalize_key(entry["metadata"]["title"], entry["metadata"]["artist"])] = entry
    _save_db(db)


def list_cached() -> list[dict]:
    return list(_load_db().values())


# ──────────────────────────────────────────────────────────────
# 음원 확보
# ──────────────────────────────────────────────────────────────

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"


def itunes_lookup(title: str, artist: str) -> Optional[dict]:
    params = {
        "term": f"{artist} {title}",
        "media": "music",
        "entity": "song",
        "limit": 5,
    }
    r = requests.get(ITUNES_SEARCH_URL, params=params, timeout=10)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        return None
    title_lc = title.strip().lower()
    for hit in results:
        if hit.get("trackName", "").strip().lower() == title_lc:
            return hit
    return results[0]


def download_itunes_preview(title: str, artist: str) -> Optional[Path]:
    hit = itunes_lookup(title, artist)
    if hit is None or not hit.get("previewUrl"):
        return None
    TMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-]+", "_", f"{artist}_{title}").strip("_")
    out = TMP_AUDIO_DIR / f"{safe}.m4a"
    if not out.exists():
        with requests.get(hit["previewUrl"], timeout=30, stream=True) as resp:
            resp.raise_for_status()
            with open(out, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
    return out


def download_youtube_audio(url: str, basename: str) -> Path:
    import yt_dlp
    TMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-]+", "_", basename).strip("_")
    out_template = str(TMP_AUDIO_DIR / f"{safe}.%(ext)s")
    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "overwrites": False,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info))


def acquire_audio(title: str, artist: str, youtube_url: Optional[str] = None) -> Path:
    """iTunes 1순위 → YouTube fallback. 둘 다 실패 시 RuntimeError."""
    p = download_itunes_preview(title, artist)
    if p is not None:
        return p
    if youtube_url:
        return download_youtube_audio(youtube_url, f"{artist}_{title}")
    raise RuntimeError(
        f"음원 확보 실패: {title} - {artist}. iTunes에 없으면 youtube_url을 넘겨주세요."
    )


# ──────────────────────────────────────────────────────────────
# Librosa feature
# ──────────────────────────────────────────────────────────────

def extract_audio_features(audio_path: str | Path) -> dict:
    import librosa

    y, sr = librosa.load(str(audio_path), mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    rms = float(np.mean(librosa.feature.rms(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    dynamics_range = float(np.percentile(onset_env, 95) - np.percentile(onset_env, 5))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key_idx = int(chroma.mean(axis=1).argmax())
    harmonic, percussive = librosa.effects.hpss(y)
    h_energy = float(np.mean(np.abs(harmonic)))
    p_energy = float(np.mean(np.abs(percussive)))
    harmonic_ratio = h_energy / (h_energy + p_energy + 1e-9)

    energy = float(min(1.0, rms * 8))
    valence = float(min(1.0, max(0.0, (centroid - 1000) / 4000)))
    danceability = float(min(1.0, max(0.0, 1.0 - zcr * 3)))

    return {
        "duration": duration,
        "tempo": tempo,
        "energy": energy,
        "valence": valence,
        "danceability": danceability,
        "key": key_idx,
        "harmonic_ratio": float(harmonic_ratio),
        "dynamics_range": dynamics_range,
        "_source": "librosa",
    }


# ──────────────────────────────────────────────────────────────
# Genius
# ──────────────────────────────────────────────────────────────

_genius_client = None


def get_genius_client():
    global _genius_client
    if _genius_client is not None:
        return _genius_client
    token, _ = load_env()
    if not token:
        raise RuntimeError("GENIUS_ACCESS_TOKEN가 .env에 없음")
    from lyricsgenius import Genius
    client = Genius(
        token,
        timeout=10,
        retries=2,
        remove_section_headers=True,
        skip_non_songs=True,
    )
    client.verbose = False
    _genius_client = client
    return _genius_client


def fetch_genius_song(title: str, artist: str):
    try:
        return get_genius_client().search_song(title, artist)
    except Exception:
        return None


def build_narrative_text(song, lyrics_chars: int = 1500) -> str:
    if song is None:
        return ""
    lyrics = re.sub(r"\n{3,}", "\n\n", (song.lyrics or "").strip())[:lyrics_chars]
    return "\n".join([
        f"Title: {song.title}",
        f"Artist: {song.artist}",
        f"Album: {getattr(song, 'album', '') or ''}",
        f"Lyrics excerpt:\n{lyrics}",
    ])


# ──────────────────────────────────────────────────────────────
# Gemini
# ──────────────────────────────────────────────────────────────

_gemini_client = None
GEMINI_TEXT_MODEL = "gemini-2.5-flash"
GEMINI_AUDIO_MODEL = "gemini-2.5-pro"


def get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    _, key = load_env()
    if not key:
        raise RuntimeError("GEMINI_API_KEY가 .env에 없음")
    from google import genai
    _gemini_client = genai.Client(api_key=key)
    return _gemini_client


def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m is None:
        raise ValueError(f"Gemini 응답에서 JSON을 찾지 못함:\n{raw}")
    return json.loads(m.group(0))


def gemini_classify_text(features: dict, narrative_text: str, title: str, artist: str) -> dict:
    from google.genai import types
    client = get_gemini_client()
    user_msg = (
        f"Song: {title} - {artist}\n"
        f"Librosa features: {json.dumps(features, ensure_ascii=False)}\n"
        f"Narrative text:\n{narrative_text}"
    )
    resp = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.3,
        ),
    )
    return _parse_json_response(resp.text)


def gemini_classify_audio(features: dict, audio_path: Path, title: str, artist: str) -> dict:
    from google.genai import types
    client = get_gemini_client()
    audio_bytes = audio_path.read_bytes()
    mime = "audio/mp4" if audio_path.suffix.lower() in {".m4a", ".mp4"} else "audio/mpeg"
    user_text = (
        f"Song: {title} - {artist}\n"
        f"Librosa features: {json.dumps(features, ensure_ascii=False)}\n"
        f"가사/메타가 없는 신곡이다. 첨부된 오디오를 직접 들어 분류하라."
    )
    resp = client.models.generate_content(
        model=GEMINI_AUDIO_MODEL,
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type=mime),
            user_text,
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.3,
        ),
    )
    return _parse_json_response(resp.text)


# ──────────────────────────────────────────────────────────────
# 라우터
# ──────────────────────────────────────────────────────────────

def process_kpop_visualization(
    title: str,
    artist: str,
    *,
    youtube_url: Optional[str] = None,
    force_refresh: bool = False,
    progress_cb=None,
) -> dict:
    """
    progress_cb(pct, msg) optional. 단계별 진행도를 통보.
    """
    if not force_refresh:
        cached = get_from_cache(title, artist)
        if cached is not None:
            if progress_cb:
                progress_cb(1.0, "캐시 히트")
            return cached

    if progress_cb:
        progress_cb(0.05, "Genius 조회")
    song = fetch_genius_song(title, artist)
    branch = "known" if song is not None else "new"

    if progress_cb:
        progress_cb(0.2, "음원 확보 (iTunes/YouTube)")
    audio_path = acquire_audio(title, artist, youtube_url=youtube_url)

    if progress_cb:
        progress_cb(0.5, "Librosa feature 추출")
    features = extract_audio_features(audio_path)

    if progress_cb:
        progress_cb(0.75, f"Gemini 분류 ({branch} 분기)")
    if branch == "known":
        narrative = build_narrative_text(song)
        visual_params = gemini_classify_text(features, narrative, title, artist)
        meta = {
            "title": title,
            "artist": artist,
            "branch": "known",
            "genius_url": getattr(song, "url", None),
            "audio_source": str(audio_path.name),
        }
    else:
        visual_params = gemini_classify_audio(features, audio_path, title, artist)
        meta = {
            "title": title,
            "artist": artist,
            "branch": "new",
            "audio_source": str(audio_path.name),
        }

    result = {
        "metadata": meta,
        "live_features": features,
        "concept_features": visual_params,
    }
    save_to_cache(result)

    if progress_cb:
        progress_cb(1.0, "완료")
    return result
