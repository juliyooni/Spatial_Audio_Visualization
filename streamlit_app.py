"""Streamlit 음원 분리 파이프라인.

사용자가 음원 파일(wav, flac, mp3, mp4 등)을 올리면
- 채널 분리 (L/R)
- 트랙 분리 (보컬 / 멜로디 악기 / 드럼)
중 원하는 것을 선택해서 뽑아주는 프론트.

실행: streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st


SUPPORTED_EXTS = [".wav", ".flac", ".mp3", ".mp4", ".m4a", ".aac", ".ogg"]


# ──────────────────────────────────────────────────────────────
# 오디오 로딩
# ──────────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """오디오를 (channels, samples) float32로 로드.

    soundfile을 먼저 시도하고, 실패하면 librosa(오디오코어) 폴백.
    mp4/aac 같은 컨테이너는 ffmpeg/audioread가 처리한다.
    """
    import librosa

    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        # soundfile: (samples, channels) → (channels, samples)
        data = data.T
    except Exception:
        y, sr = librosa.load(path, sr=None, mono=False)
        data = y if y.ndim == 2 else np.stack([y, y])

    if data.shape[0] == 1:
        data = np.concatenate([data, data], axis=0)

    if target_sr is not None and sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return data.astype(np.float32), sr


def wav_bytes(data: np.ndarray, sr: int) -> bytes:
    """(channels, samples) → WAV 바이트."""
    buf = io.BytesIO()
    sf.write(buf, data.T, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────
# 채널 분리
# ──────────────────────────────────────────────────────────────

def split_channels(data: np.ndarray) -> dict[str, np.ndarray]:
    """(channels, samples)에서 L/R 및 Mid/Side 분리."""
    left = data[0:1]
    right = data[1:2] if data.shape[0] > 1 else data[0:1]
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    return {
        "left": np.concatenate([left, left], axis=0),
        "right": np.concatenate([right, right], axis=0),
        "mid": np.concatenate([mid, mid], axis=0),
        "side": np.concatenate([side, side], axis=0),
    }


# ──────────────────────────────────────────────────────────────
# 트랙 분리 (Demucs)
# ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_demucs_model(model_name: str = "htdemucs"):
    import torch
    from demucs.pretrained import get_model

    model = get_model(model_name)
    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    return model, device


def separate_stems(data: np.ndarray, sr: int, progress_cb=None) -> dict[str, np.ndarray]:
    """Demucs 기본 4 stem으로 분리: vocals / drums / bass / other.

    - vocals: 보컬
    - drums: 드럼/퍼커션 (비트)
    - bass: 베이스
    - other: 그 외 멜로디 악기 (기타/피아노/신스 등)
    """
    import torch
    import librosa
    from demucs.apply import apply_model

    model, device = load_demucs_model()

    if sr != model.samplerate:
        data = librosa.resample(data, orig_sr=sr, target_sr=model.samplerate)
        sr = model.samplerate

    wav = torch.from_numpy(data)
    ref = wav.mean(0)
    wav_input = (wav - ref.mean()) / (ref.std() + 1e-8)

    if progress_cb:
        progress_cb(0.1, "모델 로드 완료 — 분리 시작")

    with torch.no_grad():
        sources = apply_model(
            model,
            wav_input[None].to(device),
            shifts=1,
            overlap=0.25,
            progress=False,
        )[0]

    sources = sources * ref.std() + ref.mean()
    sources = sources.cpu().numpy()

    src_idx = {name: i for i, name in enumerate(model.sources)}

    if progress_cb:
        progress_cb(1.0, "완료")

    return {
        "vocals": sources[src_idx["vocals"]],
        "drums": sources[src_idx["drums"]],
        "bass": sources[src_idx["bass"]],
        "other": sources[src_idx["other"]],
        "_sr": sr,
    }


# 프리셋 정의: 각 프리셋은 "출력 트랙 이름 → 합칠 stem 리스트" 맵
TRACK_PRESETS: dict[str, dict[str, list[str]]] = {
    # 3-track 기본 (bass는 멜로디에 포함)
    "3트랙: 보컬 / 멜로디(+베이스) / 비트": {
        "vocals": ["vocals"],
        "melody": ["bass", "other"],
        "drums": ["drums"],
    },
    # 4-track (bass 분리)
    "4트랙: 보컬 / 멜로디 / 베이스 / 비트": {
        "vocals": ["vocals"],
        "melody": ["other"],
        "bass": ["bass"],
        "drums": ["drums"],
    },
    # 단일 트랙
    "보컬만": {"vocals": ["vocals"]},
    "멜로디만 (기타/피아노/신스)": {"melody": ["other"]},
    "베이스만": {"bass": ["bass"]},
    "비트만 (드럼)": {"drums": ["drums"]},
    "반주만 (보컬 제외)": {"instrumental": ["drums", "bass", "other"]},
    # 2트랙 조합
    "보컬 + 멜로디": {"vocals": ["vocals"], "melody": ["other"]},
    "보컬 + 비트": {"vocals": ["vocals"], "drums": ["drums"]},
    "보컬 + 베이스": {"vocals": ["vocals"], "bass": ["bass"]},
    "멜로디 + 비트": {"melody": ["other"], "drums": ["drums"]},
    "멜로디 + 베이스": {"melody": ["other"], "bass": ["bass"]},
    "비트 + 베이스 (리듬 섹션)": {"drums": ["drums"], "bass": ["bass"]},
}


def apply_preset(stems: dict[str, np.ndarray], preset: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """stem dict + 프리셋 → 출력 트랙 dict."""
    out = {}
    for out_name, stem_list in preset.items():
        mixed = sum(stems[s] for s in stem_list)
        out[out_name] = mixed
    return out


# ──────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Audio Splitter", page_icon="🎧", layout="wide")

st.title("🎧 Audio Splitter")
st.caption("음원 파일을 올리면 채널 또는 트랙 단위로 분리해줍니다.")

# ─── 업로드 ───
uploaded = st.file_uploader(
    "음원 파일 업로드",
    type=[ext.lstrip(".") for ext in SUPPORTED_EXTS],
    help=f"지원 포맷: {', '.join(SUPPORTED_EXTS)}",
)

if uploaded is None:
    st.info("음원 파일을 업로드하세요.")
    st.stop()

# 업로드 파일을 임시 경로에 저장
ext = Path(uploaded.name).suffix.lower()
with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
    tmp.write(uploaded.read())
    input_path = tmp.name

stem_name = Path(uploaded.name).stem

with st.spinner("오디오 로딩 중..."):
    try:
        audio, sr = load_audio(input_path)
    except Exception as e:
        st.error(f"오디오 로딩 실패: {e}")
        st.stop()

duration = audio.shape[1] / sr
col_a, col_b, col_c = st.columns(3)
col_a.metric("길이", f"{duration:.1f} s")
col_b.metric("샘플레이트", f"{sr} Hz")
col_c.metric("채널", audio.shape[0])

st.audio(uploaded.getvalue() if hasattr(uploaded, "getvalue") else input_path)

st.divider()

# ─── 분리 모드 선택 ───
mode = st.radio(
    "분리 방식",
    ["채널 분리 (L / R / Mid / Side)", "트랙 분리 (보컬 / 멜로디 / 비트)"],
    horizontal=True,
)

# ─── 채널 분리 ───
if mode.startswith("채널"):
    st.subheader("채널 분리")
    channel_opts = st.multiselect(
        "추출할 채널 선택",
        ["left", "right", "mid", "side"],
        default=["left", "right"],
        format_func=lambda x: {
            "left": "Left (L)",
            "right": "Right (R)",
            "mid": "Mid (L+R)/2",
            "side": "Side (L-R)/2",
        }[x],
    )

    if st.button("분리 실행", type="primary", disabled=len(channel_opts) == 0):
        splits = split_channels(audio)
        selected = {k: splits[k] for k in channel_opts}
        st.session_state["results"] = selected
        st.session_state["results_sr"] = sr
        st.session_state["stem_name"] = stem_name
        st.session_state["mode"] = "channel"

# ─── 트랙 분리 ───
else:
    st.subheader("트랙 분리")
    st.caption("Meta의 Demucs v4 (htdemucs) 모델 사용. 첫 실행 시 모델 다운로드(~80MB).")

    preset_name = st.selectbox(
        "추출할 트랙 조합 선택",
        list(TRACK_PRESETS.keys()),
        index=0,
        help="Demucs는 보컬/드럼/베이스/기타 외 멜로디 악기(other) 4가지로만 분리됩니다. "
             "아래 조합 중에서 원하는 출력 묶음을 고르세요.",
    )

    preset = TRACK_PRESETS[preset_name]

    # 선택한 프리셋 구성 미리보기
    with st.expander("이 프리셋이 뽑아주는 트랙", expanded=False):
        for out_name, stem_list in preset.items():
            stem_labels = {
                "vocals": "보컬",
                "drums": "드럼",
                "bass": "베이스",
                "other": "기타/피아노/신스 등",
            }
            desc = " + ".join(stem_labels[s] for s in stem_list)
            st.markdown(f"- **{out_name}**: {desc}")

    if st.button("분리 실행", type="primary"):
        prog = st.progress(0.0, text="시작")

        def cb(pct, msg):
            prog.progress(pct, text=msg)

        with st.spinner("트랙 분리 중... (곡 길이에 따라 수십 초~수 분)"):
            try:
                stems = separate_stems(audio, sr, progress_cb=cb)
            except Exception as e:
                st.error(f"분리 실패: {e}")
                st.stop()

        out_sr = stems.pop("_sr")
        selected = apply_preset(stems, preset)
        st.session_state["results"] = selected
        st.session_state["results_sr"] = out_sr
        st.session_state["stem_name"] = stem_name
        st.session_state["mode"] = "track"
        prog.empty()

# ─── 결과 표시 ───
if "results" in st.session_state:
    st.divider()
    st.subheader("결과")

    results = st.session_state["results"]
    out_sr = st.session_state["results_sr"]
    stem = st.session_state["stem_name"]
    mode_key = st.session_state["mode"]

    label_map = {
        "left": "Left", "right": "Right", "mid": "Mid", "side": "Side",
        "vocals": "Vocals (보컬)",
        "melody": "Melody (멜로디 악기)",
        "drums": "Drums (비트)",
        "bass": "Bass (베이스)",
        "instrumental": "Instrumental (반주 — 보컬 제외)",
    }

    # 각 트랙을 개별 플레이어 + 다운로드
    for key, data in results.items():
        st.markdown(f"**{label_map.get(key, key)}**")
        audio_bytes = wav_bytes(data, out_sr)
        col1, col2 = st.columns([4, 1])
        with col1:
            st.audio(audio_bytes, format="audio/wav")
        with col2:
            st.download_button(
                "⬇ WAV 다운로드",
                data=audio_bytes,
                file_name=f"{stem}_{key}.wav",
                mime="audio/wav",
                key=f"dl_{key}",
            )

    # 전체 zip 다운로드
    if len(results) > 1:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for key, data in results.items():
                zf.writestr(f"{stem}_{key}.wav", wav_bytes(data, out_sr))
        st.download_button(
            "📦 전체 ZIP 다운로드",
            data=zip_buf.getvalue(),
            file_name=f"{stem}_{mode_key}_stems.zip",
            mime="application/zip",
            type="primary",
        )

# 임시 파일 정리
try:
    os.unlink(input_path)
except OSError:
    pass
