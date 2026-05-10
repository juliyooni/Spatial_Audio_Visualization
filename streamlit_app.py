"""Spatial Audio Visualization — K-pop 통합 분석 Streamlit 앱.

메인 페이지에서 (음원 파일 + 곡 제목 + 아티스트)를 한 번 입력하면 체크된
파이프라인이 순차 실행되고, 결과는 탭에서 조회한다:
  - 🎼 음악 분석 — 피치 / 구조 경계 / 분위기 (Librosa)
  - 🎧 음원 분리 — 채널 / 트랙(Demucs) — 결과 탭에서 프리셋 변경 가능
  - 🌌 Visual Mapping — Librosa + Genius + Gemini로 비주얼 컨셉 JSON
  - 📚 캐시 DB — 과거 Visual Mapping 결과 열람

실행: streamlit run streamlit_app.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components

# 모듈 import 가능하도록 PROJECT_ROOT를 sys.path에 추가
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


SUPPORTED_EXTS = [".wav", ".flac", ".mp3", ".mp4", ".m4a", ".aac", ".ogg"]


# ══════════════════════════════════════════════════════════════
# 공용 유틸 — 오디오 로딩, WAV 직렬화
# ══════════════════════════════════════════════════════════════

def load_audio_stereo(path: str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """(channels, samples) float32. 음원 분리 페이지에서 사용."""
    import librosa

    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
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
    if data.ndim == 1:
        sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    else:
        sf.write(buf, data.T, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def save_uploaded(uploaded) -> str:
    """업로드 파일을 임시 경로에 저장하고 경로 반환."""
    ext = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded.read())
        return tmp.name


# ══════════════════════════════════════════════════════════════
# 페이지 1 — 음원 분리 (채널 / 트랙)
# ══════════════════════════════════════════════════════════════

def split_channels(data: np.ndarray) -> dict[str, np.ndarray]:
    """L/R은 실제 출력 채널에만 보내고, Mid/Side는 듀얼 모노로 양쪽에 보낸다.

    - left  → [L, 0]  : 왼쪽 스피커에서만
    - right → [0, R]  : 오른쪽 스피커에서만
    - mid   → [M, M]  : (L+R)/2를 양쪽 모두 (가운데 정위)
    - side  → [S, S]  : (L-R)/2를 양쪽 모두 (Mid/Side는 좌우 개념이 없음)
    """
    left = data[0:1]
    right = data[1:2] if data.shape[0] > 1 else data[0:1]
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    silence = np.zeros_like(left)
    return {
        "left": np.concatenate([left, silence], axis=0),
        "right": np.concatenate([silence, right], axis=0),
        "mid": np.concatenate([mid, mid], axis=0),
        "side": np.concatenate([side, side], axis=0),
    }


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


TRACK_PRESETS: dict[str, dict[str, list[str]]] = {
    "3트랙: 보컬 / 멜로디(+베이스) / 비트": {
        "vocals": ["vocals"],
        "melody": ["bass", "other"],
        "drums": ["drums"],
    },
    "4트랙: 보컬 / 멜로디 / 베이스 / 비트": {
        "vocals": ["vocals"],
        "melody": ["other"],
        "bass": ["bass"],
        "drums": ["drums"],
    },
    "보컬만": {"vocals": ["vocals"]},
    "멜로디만 (기타/피아노/신스)": {"melody": ["other"]},
    "베이스만": {"bass": ["bass"]},
    "비트만 (드럼)": {"drums": ["drums"]},
    "반주만 (보컬 제외)": {"instrumental": ["drums", "bass", "other"]},
    "보컬 + 멜로디": {"vocals": ["vocals"], "melody": ["other"]},
    "보컬 + 비트": {"vocals": ["vocals"], "drums": ["drums"]},
    "보컬 + 베이스": {"vocals": ["vocals"], "bass": ["bass"]},
    "멜로디 + 비트": {"melody": ["other"], "drums": ["drums"]},
    "멜로디 + 베이스": {"melody": ["other"], "bass": ["bass"]},
    "비트 + 베이스 (리듬 섹션)": {"drums": ["drums"], "bass": ["bass"]},
}


def apply_preset(stems: dict[str, np.ndarray], preset: dict[str, list[str]]) -> dict[str, np.ndarray]:
    out = {}
    for out_name, stem_list in preset.items():
        out[out_name] = sum(stems[s] for s in stem_list)
    return out


def tab_audio_splitter():
    """음원 분리 결과 탭. 미리 계산된 채널/Demucs stem을 session에서 읽는다."""
    bundle = st.session_state.get("run_bundle", {})
    splitter = bundle.get("splitter")
    if not splitter:
        st.info("음원 분리가 실행되지 않았습니다. 메인 페이지에서 '음원 분리'를 체크하고 실행하세요.")
        return

    audio = splitter["audio"]
    sr = splitter["sr"]
    stem_name = splitter["stem"]
    demucs_stems = splitter.get("demucs_stems")  # None이면 트랙 분리는 skip된 상태
    out_sr = splitter.get("demucs_sr") or sr

    duration = audio.shape[1] / sr
    c1, c2, c3 = st.columns(3)
    c1.metric("길이", f"{duration:.1f} s")
    c2.metric("샘플레이트", f"{sr} Hz")
    c3.metric("채널", audio.shape[0])

    st.divider()

    mode_options = ["채널 분리 (L / R / Mid / Side)"]
    if demucs_stems is not None:
        mode_options.append("트랙 분리 (보컬 / 멜로디 / 비트)")
    mode = st.radio(
        "분리 방식",
        mode_options,
        horizontal=True,
        key="splitter_mode",
    )

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
        if not channel_opts:
            st.info("채널을 하나 이상 선택하세요.")
            return
        splits = split_channels(audio)
        results = {k: splits[k] for k in channel_opts}
        results_sr = sr
        mode_key = "channel"

    else:
        st.subheader("트랙 분리")
        st.caption("Meta Demucs v4 (htdemucs). 결과는 미리 계산되어 있어 프리셋만 바꾸면 즉시 합쳐집니다.")
        preset_name = st.selectbox(
            "추출할 트랙 조합 선택",
            list(TRACK_PRESETS.keys()),
            index=0,
        )
        preset = TRACK_PRESETS[preset_name]
        with st.expander("이 프리셋이 뽑아주는 트랙", expanded=False):
            stem_labels = {
                "vocals": "보컬", "drums": "드럼", "bass": "베이스",
                "other": "기타/피아노/신스 등",
            }
            for out_name, stem_list in preset.items():
                desc = " + ".join(stem_labels[s] for s in stem_list)
                st.markdown(f"- **{out_name}**: {desc}")
        results = apply_preset(demucs_stems, preset)
        results_sr = out_sr
        mode_key = "track"

    st.divider()
    st.subheader("결과")

    label_map = {
        "left": "Left", "right": "Right", "mid": "Mid", "side": "Side",
        "vocals": "Vocals (보컬)",
        "melody": "Melody (멜로디 악기)",
        "drums": "Drums (비트)",
        "bass": "Bass (베이스)",
        "instrumental": "Instrumental (반주 — 보컬 제외)",
    }

    for key, data in results.items():
        st.markdown(f"**{label_map.get(key, key)}**")
        audio_out = wav_bytes(data, results_sr)
        col1, col2 = st.columns([4, 1])
        with col1:
            st.audio(audio_out, format="audio/wav")
        with col2:
            st.download_button(
                "⬇ WAV",
                data=audio_out,
                file_name=f"{stem_name}_{key}.wav",
                mime="audio/wav",
                key=f"dl_split_{mode_key}_{key}",
            )

    if len(results) > 1:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for key, data in results.items():
                zf.writestr(f"{stem_name}_{key}.wav", wav_bytes(data, results_sr))
        st.download_button(
            "📦 전체 ZIP 다운로드",
            data=zip_buf.getvalue(),
            file_name=f"{stem_name}_{mode_key}_stems.zip",
            mime="application/zip",
            type="primary",
            key=f"dl_zip_{mode_key}",
        )


# ══════════════════════════════════════════════════════════════
# 페이지 2 — 음악 분석 (피치 / 구조 / 분위기)
# ══════════════════════════════════════════════════════════════

def _plot_analysis(result: dict):
    """matplotlib 3행 플롯 — 파형/섹션/분위기/피치/노벨티."""
    import matplotlib.pyplot as plt
    import librosa.display

    y = result["_audio"]["mono"]
    sr = result["_audio"]["sr"]
    pitch = result["pitch"]
    sections = result["mood"]["sections"]
    struct_bounds = result["segmentation"]["boundary_times"]
    mood_bounds = result["mood"]["mood_boundaries"]["times"]
    novelty = result["mood"]["mood_boundaries"]["novelty"]
    nov_times = result["mood"]["mood_boundaries"]["frame_times"]
    duration = result["duration_sec"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    librosa.display.waveshow(y, sr=sr, ax=axes[0], alpha=0.5)
    axes[0].set_title(
        f"{Path(result['path']).name} · {duration:.1f}s · {result['tempo_bpm']} BPM   "
        "(red --: structural, blue ··: mood)"
    )
    axes[0].set_ylabel("amp")

    cmap = plt.get_cmap("tab20")
    for s in sections:
        color = cmap(s["section_idx"] % 20)
        axes[0].axvspan(s["start"], s["end"], color=color, alpha=0.12)
        mid = (s["start"] + s["end"]) / 2
        axes[0].text(
            mid, axes[0].get_ylim()[1] * 0.85,
            f"#{s['section_idx']}\n{s['mood_label']}",
            ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
        )
    for b in struct_bounds:
        axes[0].axvline(b, color="crimson", lw=1.2, ls="--", alpha=0.8)
    for b in mood_bounds:
        axes[0].axvline(b, color="royalblue", lw=1.2, ls=":", alpha=0.9)

    axes[1].plot(pitch["times"], pitch["f0_hz"], lw=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("f0 (Hz)")
    axes[1].set_title("Pitch contour (pYIN)")
    for b in struct_bounds:
        axes[1].axvline(b, color="crimson", lw=1, ls="--", alpha=0.4)
    for b in mood_bounds:
        axes[1].axvline(b, color="royalblue", lw=1, ls=":", alpha=0.5)

    axes[2].plot(nov_times, novelty, color="royalblue", lw=1)
    axes[2].set_ylabel("mood novelty")
    axes[2].set_xlabel("time (s)")
    axes[2].set_title("Mood novelty curve")
    for b in mood_bounds:
        axes[2].axvline(b, color="royalblue", lw=1, ls=":", alpha=0.6)

    tick_step = 5 if duration < 120 else 10
    axes[2].set_xticks(np.arange(0, duration + 1, tick_step))
    axes[2].set_xlim(0, duration)
    axes[2].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


def _plotly_analysis(result: dict, show_struct: bool, show_mood: bool, show_unified: bool = False):
    """Plotly 3행 인터랙티브 차트 — 파형(다운샘플) / 피치 / novelty.

    matplotlib 백업 플롯과 같은 정보를 보여주되 줌·팬·호버가 가능하도록.
    재생 위치 동기화는 별도의 WaveSurfer 위젯에서 처리.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    y = result["_audio"]["mono"]
    sr = result["_audio"]["sr"]
    pitch = result["pitch"]
    struct_bounds = result["segmentation"]["boundary_times"]
    mood_bounds = result["mood"]["mood_boundaries"]["times"]
    novelty = result["mood"]["mood_boundaries"]["novelty"]
    nov_times = result["mood"]["mood_boundaries"]["frame_times"]
    duration = result["duration_sec"]

    # 통합 모드면 통합 섹션과 경계를 사용
    unified = result.get("unified", {})
    unified_bounds = unified.get("boundary_times", []) if show_unified else []
    if show_unified and unified.get("sections"):
        sections = unified["sections"]
    else:
        sections = result["mood"]["sections"]

    # 파형은 픽셀 수 정도로 다운샘플 — 5분 곡도 가볍게 그릴 수 있게
    target_pts = 4000
    if len(y) > target_pts:
        step = len(y) // target_pts
        wf_y = y[::step]
        wf_t = np.arange(len(wf_y)) * step / sr
    else:
        wf_y = y
        wf_t = np.arange(len(wf_y)) / sr

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=("Waveform + sections", "Pitch contour (pYIN)", "Mood novelty"),
    )

    # 섹션 음영 (구조 경계 기준)
    palette = [
        "rgba(31,119,180,0.10)", "rgba(255,127,14,0.10)", "rgba(44,160,44,0.10)",
        "rgba(214,39,40,0.10)", "rgba(148,103,189,0.10)", "rgba(140,86,75,0.10)",
        "rgba(227,119,194,0.10)", "rgba(127,127,127,0.10)", "rgba(188,189,34,0.10)",
        "rgba(23,190,207,0.10)",
    ]
    for s in sections:
        fig.add_vrect(
            x0=s["start"], x1=s["end"],
            fillcolor=palette[s["section_idx"] % len(palette)],
            line_width=0, layer="below",
            row=1, col=1,
        )
        fig.add_annotation(
            x=(s["start"] + s["end"]) / 2,
            y=1, yref="y domain",
            text=f"#{s['section_idx']} · {s['mood_label']}",
            showarrow=False, yanchor="top",
            font=dict(size=10, color="#444"),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="#bbb", borderwidth=1,
            row=1, col=1,
        )

    # 파형
    fig.add_trace(
        go.Scatter(
            x=wf_t, y=wf_y, mode="lines",
            line=dict(color="#888", width=1),
            name="waveform", hoverinfo="skip", showlegend=False,
        ),
        row=1, col=1,
    )

    # 피치
    f0 = pitch["f0_hz"]
    fig.add_trace(
        go.Scatter(
            x=pitch["times"], y=f0, mode="lines",
            line=dict(color="#1f77b4", width=1),
            name="f0", hovertemplate="t=%{x:.2f}s<br>f0=%{y:.1f} Hz<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )
    valid = f0[~np.isnan(f0)] if f0.size else f0
    if valid.size:
        fig.update_yaxes(type="log", row=2, col=1)

    # Mood novelty
    fig.add_trace(
        go.Scatter(
            x=nov_times, y=novelty, mode="lines",
            line=dict(color="#4169e1", width=1.4),
            name="novelty",
            hovertemplate="t=%{x:.2f}s<br>novelty=%{y:.2f}<extra></extra>",
            showlegend=False,
            fill="tozeroy", fillcolor="rgba(65,105,225,0.10)",
        ),
        row=3, col=1,
    )

    # 경계선 — 모든 row에 vline
    if show_struct:
        for b in struct_bounds:
            fig.add_vline(x=b, line=dict(color="crimson", width=1.2, dash="dash"), opacity=0.7)
    if show_mood:
        for b in mood_bounds:
            fig.add_vline(x=b, line=dict(color="royalblue", width=1.2, dash="dot"), opacity=0.8)
    if show_unified:
        unified_sources = unified.get("boundary_sources", [])
        for i, b in enumerate(unified_bounds):
            fig.add_vline(x=b, line=dict(color="seagreen", width=1.5), opacity=0.85)
            srcs = unified_sources[i] if i < len(unified_sources) else []
            if "struct" in srcs and "mood" in srcs:
                tag = "S+M"
            elif "struct" in srcs:
                tag = "S"
            elif "mood" in srcs:
                tag = "M"
            else:
                continue  # start/end는 라벨 안 붙임
            # 첫 번째 row 상단에 작은 라벨
            fig.add_annotation(
                x=b, y=1.02, yref="y domain", xref="x",
                text=tag, showarrow=False,
                font=dict(size=9, color="seagreen", family="monospace"),
                row=1, col=1,
            )

    fig.update_layout(
        height=620, margin=dict(l=50, r=20, t=40, b=40),
        hovermode="x unified",
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)", range=[0, duration])
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(title_text="amp", row=1, col=1)
    fig.update_yaxes(title_text="f0 (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="novelty", row=3, col=1)
    fig.update_xaxes(title_text="time (s)", row=3, col=1)
    return fig


def _wavesurfer_player(
    audio_bytes: bytes,
    audio_mime: str,
    duration: float,
    struct_bounds: list[float],
    mood_bounds: list[float],
    sections: list[dict],
    show_struct: bool,
    show_mood: bool,
    unified_bounds: list[float] | None = None,
    unified_sources: list[list[str]] | None = None,
    show_unified: bool = False,
    view_title: str | None = None,
    height: int = 220,
):
    """WaveSurfer.js로 파형 + 재생 커서 + 경계 region을 그린다.

    오디오는 base64 data URL로 임베드 (Streamlit components iframe은 별도 origin이라
    파일 시스템 접근 불가). 곡이 길수록 페이로드가 커지므로 K-pop 길이(3~4분)에선 OK.
    """
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    data_url = f"data:{audio_mime};base64,{b64}"

    # 통합 마커에 표시할 소스 라벨 — "S"=구조, "M"=분위기, "S+M"=합쳐진 것
    def _src_label(srcs: list[str]) -> str:
        has_s = "struct" in srcs
        has_m = "mood" in srcs
        if has_s and has_m:
            return "S+M"
        if has_s:
            return "S"
        if has_m:
            return "M"
        return ""  # start/end는 라벨 없음

    unified_payload = []
    if show_unified and unified_bounds:
        srcs = unified_sources or [[] for _ in unified_bounds]
        for t, src in zip(unified_bounds, srcs):
            unified_payload.append({"t": float(t), "label": _src_label(src)})

    # JS에 넘길 데이터
    payload = {
        "audio": data_url,
        "duration": duration,
        "struct": [float(b) for b in struct_bounds] if show_struct else [],
        "mood": [float(b) for b in mood_bounds] if show_mood else [],
        "unified": unified_payload,
        "sections": [
            {
                "start": float(s["start"]),
                "end": float(s["end"]),
                "idx": int(s["section_idx"]),
                "label": str(s["mood_label"]),
            }
            for s in sections
        ],
        "view_title": view_title or "",
        "show_struct": bool(show_struct),
        "show_mood": bool(show_mood),
        "show_unified": bool(show_unified),
    }

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #222; }}
  #wrap {{ padding: 2px 4px; }}
  .view-title {{ font-size: 14px; font-weight: 600; margin: 0 0 4px 0; color: #222; }}
  #waveform {{ background: #f7f7f7; border-radius: 6px; }}
  .controls {{ display: flex; gap: 8px; align-items: center; margin-top: 6px; }}
  .controls button {{
    background: #1f77b4; color: white; border: none; border-radius: 4px;
    padding: 4px 12px; cursor: pointer; font-size: 12px;
  }}
  .controls button:hover {{ background: #145a8a; }}
  .controls button.secondary {{ background: #888; }}
  .controls button.secondary:hover {{ background: #555; }}
  #time {{ font-variant-numeric: tabular-nums; font-size: 12px; color: #444; margin-left: auto; }}
  .legend {{ font-size: 11px; color: #666; margin-top: 4px; }}
  .legend .sw {{ display: inline-block; width: 10px; height: 10px; vertical-align: middle; margin-right: 3px; border-radius: 2px; }}
  #section-bar {{
    margin-top: 6px; display: flex; gap: 2px;
    font-size: 10px; color: #fff;
  }}
  #section-bar .seg {{
    padding: 3px 6px; border-radius: 3px; cursor: pointer;
    overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
    text-shadow: 0 0 2px rgba(0,0,0,0.6);
  }}
</style>
</head>
<body>
<div id="wrap">
  <div id="view-title" class="view-title"></div>
  <div id="waveform"></div>
  <div id="section-bar"></div>
  <div class="controls">
    <button id="play">▶ Play</button>
    <button id="restart" class="secondary">⏮ 처음으로</button>
    <span id="time">0:00 / 0:00</span>
  </div>
  <div id="legend" class="legend"></div>
</div>

<script src="https://unpkg.com/wavesurfer.js@7.8.6/dist/wavesurfer.min.js"></script>
<script src="https://unpkg.com/wavesurfer.js@7.8.6/dist/plugins/regions.min.js"></script>
<script src="https://unpkg.com/wavesurfer.js@7.8.6/dist/plugins/timeline.min.js"></script>
<script>
  const DATA = {json.dumps(payload)};

  const palette = [
    "rgba(31,119,180,0.55)", "rgba(255,127,14,0.55)", "rgba(44,160,44,0.55)",
    "rgba(214,39,40,0.55)",  "rgba(148,103,189,0.55)","rgba(140,86,75,0.55)",
    "rgba(227,119,194,0.55)","rgba(127,127,127,0.55)","rgba(188,189,34,0.55)",
    "rgba(23,190,207,0.55)",
  ];

  const ws = WaveSurfer.create({{
    container: '#waveform',
    waveColor: '#bbb',
    progressColor: '#1f77b4',
    cursorColor: '#d62728',
    cursorWidth: 2,
    height: 100,
    normalize: true,
    barWidth: 2,
    barGap: 1,
    barRadius: 1,
    plugins: [
      WaveSurfer.Timeline.create({{
        container: '#waveform',
        primaryLabelInterval: 10,
        secondaryLabelInterval: 5,
        style: 'font-size: 10px; color: #777;',
      }}),
    ],
  }});

  const regions = ws.registerPlugin(WaveSurfer.Regions.create());

  ws.load(DATA.audio);

  ws.on('decode', () => {{
    // 섹션 region (음영 + 라벨, 클릭하면 점프)
    DATA.sections.forEach((s, i) => {{
      regions.addRegion({{
        start: s.start,
        end: s.end,
        color: palette[s.idx % palette.length].replace('0.55', '0.10'),
        drag: false, resize: false,
        content: '#' + s.idx + ' ' + s.label,
      }});
    }});
    // 구조 경계 — 빨간 마커
    DATA.struct.forEach(t => {{
      regions.addRegion({{
        start: t, end: t,
        color: 'rgba(220,20,60,0.9)',
        drag: false, resize: false,
      }});
    }});
    // 분위기 경계 — 파란 마커
    DATA.mood.forEach(t => {{
      regions.addRegion({{
        start: t, end: t,
        color: 'rgba(65,105,225,0.9)',
        drag: false, resize: false,
      }});
    }});
    // 통합 경계 — 녹색 마커 + 소스 라벨(S/M/S+M)
    DATA.unified.forEach(o => {{
      regions.addRegion({{
        start: o.t, end: o.t,
        color: 'rgba(46,139,87,0.95)',
        drag: false, resize: false,
        content: o.label || '',
      }});
    }});

    // 영역 클릭 — 클릭한 정확한 픽셀 위치를 시간으로 환산해서 점프
    regions.on('region-clicked', (region, e) => {{
      e.stopPropagation();
      const wf = document.getElementById('waveform');
      const rect = wf.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const ratio = Math.max(0, Math.min(1, x / rect.width));
      ws.setTime(ratio * DATA.duration);
    }});

    renderSectionBar();
    renderLegend();
    renderTitle();
  }});

  function fmt(s) {{
    if (!isFinite(s)) return '0:00';
    const m = Math.floor(s / 60);
    const r = Math.floor(s % 60);
    return m + ':' + (r < 10 ? '0' : '') + r;
  }}

  const timeEl = document.getElementById('time');
  ws.on('timeupdate', t => {{
    timeEl.textContent = fmt(t) + ' / ' + fmt(DATA.duration);
  }});
  ws.on('ready', () => {{
    timeEl.textContent = '0:00 / ' + fmt(DATA.duration);
  }});

  const playBtn = document.getElementById('play');
  playBtn.onclick = () => ws.playPause();
  ws.on('play', () => playBtn.textContent = '⏸ Pause');
  ws.on('pause', () => playBtn.textContent = '▶ Play');
  ws.on('finish', () => playBtn.textContent = '▶ Play');

  document.getElementById('restart').onclick = () => {{
    ws.setTime(0);
    if (!ws.isPlaying()) ws.play();
  }};

  // 섹션 바 — 클릭한 정확한 위치(섹션 내 비율)로 점프
  function renderSectionBar() {{
    const bar = document.getElementById('section-bar');
    bar.innerHTML = '';
    DATA.sections.forEach(s => {{
      const seg = document.createElement('div');
      seg.className = 'seg';
      seg.style.background = palette[s.idx % palette.length].replace('0.55', '0.85');
      seg.style.flex = (s.end - s.start);
      seg.title = '#' + s.idx + ' · ' + s.label + ' (' + s.start.toFixed(1) + '–' + s.end.toFixed(1) + 's)';
      seg.textContent = '#' + s.idx + ' ' + s.label;
      seg.onclick = (e) => {{
        const rect = seg.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        ws.setTime(s.start + ratio * (s.end - s.start));
      }};
      bar.appendChild(seg);
    }});
  }}

  function renderTitle() {{
    const el = document.getElementById('view-title');
    if (DATA.view_title) {{
      el.textContent = '🔍 ' + DATA.view_title;
    }} else {{
      el.style.display = 'none';
    }}
  }}

  function renderLegend() {{
    const el = document.getElementById('legend');
    const parts = [];
    if (DATA.show_struct) {{
      parts.push('<span class="sw" style="background:crimson"></span>구조 경계');
    }}
    if (DATA.show_mood) {{
      parts.push('<span class="sw" style="background:royalblue"></span>분위기 경계');
    }}
    if (DATA.show_unified) {{
      parts.push('<span class="sw" style="background:seagreen"></span>통합 (S=구조 / M=분위기 / S+M=합쳐진 것)');
    }}
    parts.push('파형/섹션을 클릭하면 그 지점으로 점프');
    el.innerHTML = parts.join(' &nbsp;·&nbsp; ');
  }}
</script>
</body>
</html>
"""
    # waveform(100) + section-bar(~22) + controls(~28) + legend(~18) + title(~22) + padding/margin
    components.html(html, height=height, scrolling=False)


def tab_music_analysis():
    """음악 분석 결과 탭. 미리 계산된 result를 session에서 읽는다."""
    from music_analysis.pipeline import export_result

    bundle = st.session_state.get("run_bundle", {})
    result = bundle.get("analysis")
    if result is None:
        st.info("음악 분석이 실행되지 않았습니다. 메인 페이지에서 '음악 분석'을 체크하고 실행하세요.")
        return

    audio_bytes = bundle.get("audio_bytes")
    audio_mime = bundle.get("audio_mime") or "audio/wav"
    filename = bundle.get("filename") or Path(result.get("path", "audio.wav")).name

    st.subheader("개요")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("길이", f"{result['duration_sec']:.1f} s")
    c2.metric("Tempo", f"{result['tempo_bpm']:.1f} BPM")
    ps = result["pitch"]["summary"]
    if ps:
        c3.metric("피치 중앙값", f"{ps['f0_median_hz']:.0f} Hz")
        c4.metric("Voiced 비율", f"{ps['voiced_ratio']*100:.0f} %")

    # ─── 경계 토글 (다중 선택) ───
    st.subheader("비교할 뷰 선택")
    VIEW_OPTIONS = ["구조 경계", "분위기 경계", "통합 분석"]
    selected_views = st.multiselect(
        "어떤 뷰를 비교하시겠어요?",
        VIEW_OPTIONS,
        default=VIEW_OPTIONS,
        key="analysis_view_selection",
        label_visibility="collapsed",
        help="여러 개 선택하면 선택된 뷰가 세로로 차례로 표시되어 같은 곡을 다른 기준으로 동시에 비교할 수 있어요.",
    )

    if not selected_views:
        st.info("최소 한 개의 뷰를 선택하세요.")
        return

    unified = result.get("unified", {})
    if "통합 분석" in selected_views and not unified.get("boundary_times"):
        st.warning(
            "통합 경계가 비어 있습니다. 분석 결과에 `unified` 키가 없거나 비었어요 — "
            "분석 버튼을 다시 눌러 재실행하면 채워집니다."
        )

    # ─── 뷰별로 차례 렌더링 (사이 공백 최소화) ───
    # components iframe들 사이의 vertical 간격을 강제로 조여 세 개가 딱 붙어 보이게.
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"] > div:has(> iframe) { margin-bottom: -8px !important; }
        iframe { display: block; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 도입부 0~3초 안쪽 경계는 어떤 뷰에서도 그리지 않는다.
    # (캐시된 옛 result에는 unified에 head_skip이 안 적용돼 있을 수 있어 여기서 한 번 더 거름)
    HEAD_SKIP_SEC = 3.0
    struct_bounds_clean = [b for b in result["segmentation"]["boundary_times"] if b >= HEAD_SKIP_SEC]
    mood_bounds_clean = [b for b in result["mood"]["mood_boundaries"]["times"] if b >= HEAD_SKIP_SEC]
    unified_filtered: list[float] = []
    unified_filtered_srcs: list[list[str]] = []
    for t, srcs in zip(unified.get("boundary_times", []), unified.get("boundary_sources", [])):
        # 곡 시작 0.0(start)은 살리고, 그 외에 도입부 안쪽이면 드롭.
        if t == 0.0 or t >= HEAD_SKIP_SEC:
            unified_filtered.append(t)
            unified_filtered_srcs.append(srcs)

    for idx, view in enumerate(selected_views):
        # "구조 경계" → 빨강만, "분위기 경계" → 파랑만, "통합 분석" → 셋 다 표시
        is_unified_view = view == "통합 분석"
        show_struct = view == "구조 경계" or is_unified_view
        show_mood = view == "분위기 경계" or is_unified_view
        show_unified = is_unified_view

        if is_unified_view and unified.get("sections"):
            ws_sections = unified["sections"]
        else:
            ws_sections = result["mood"]["sections"]

        try:
            _wavesurfer_player(
                audio_bytes=audio_bytes,
                audio_mime=audio_mime,
                duration=float(result["duration_sec"]),
                struct_bounds=struct_bounds_clean,
                mood_bounds=mood_bounds_clean,
                sections=ws_sections,
                show_struct=show_struct,
                show_mood=show_mood,
                unified_bounds=unified_filtered,
                unified_sources=unified_filtered_srcs,
                show_unified=show_unified,
                view_title=view,
            )
        except Exception as e:
            st.warning(f"플레이어 로드 실패: {e}")
            if audio_bytes:
                st.audio(audio_bytes)

    # ─── 공통: matplotlib 백업 ───
    st.divider()
    with st.expander("정적 리포트 (matplotlib)", expanded=False):
        try:
            mpl_fig = _plot_analysis(result)
            st.pyplot(mpl_fig, clear_figure=True)
        except Exception as e:
            st.warning(f"플롯 생성 실패: {e}")

    st.divider()
    st.subheader("내보내기")
    if unified.get("sections"):
        st.caption(
            "L/R wav + 통합 sections/pitch/mood_frames/boundaries/meta CSV를 ZIP으로 묶어 다운로드. "
            "sections·boundaries는 통합 분석 결과가 들어갑니다 (boundary_sources=struct/mood/struct+mood)."
        )
    else:
        st.caption(
            "L/R wav + sections/pitch/mood_frames/boundaries/meta CSV를 ZIP으로 묶어 다운로드. "
            "⚠️ 통합 분석 결과가 비어 있어 mood 섹션과 raw 경계로 폴백합니다 — 분석을 다시 돌리세요."
        )

    if st.button("ZIP 생성", key="analysis_export"):
        with st.spinner("CSV/WAV 생성 중..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                # export_result는 result["path"]의 stem을 파일명으로 사용.
                # 업로드 파일명으로 보이도록 path 임시 교체.
                orig_path = result["path"]
                pretty_stem = Path(filename).stem
                fake_path = os.path.join(tmpdir, f"{pretty_stem}{Path(orig_path).suffix}")
                result["path"] = fake_path
                try:
                    out_dir = export_result(result, out_dir=tmpdir)
                finally:
                    result["path"] = orig_path

                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fp in Path(out_dir).iterdir():
                        zf.write(fp, arcname=fp.name)
                st.session_state["analysis_zip"] = zip_buf.getvalue()
                st.session_state["analysis_zip_name"] = f"{pretty_stem}_analysis.zip"

    if "analysis_zip" in st.session_state:
        st.download_button(
            "📦 분석 결과 ZIP 다운로드",
            data=st.session_state["analysis_zip"],
            file_name=st.session_state["analysis_zip_name"],
            mime="application/zip",
            type="primary",
        )


# ══════════════════════════════════════════════════════════════
# 페이지 3 — K-pop Visual Mapping (Genius + Gemini)
# ══════════════════════════════════════════════════════════════

def _render_visual_concept(result: dict):
    meta = result["metadata"]
    feats = result["live_features"]
    concept = result["concept_features"]

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(f"### {meta['title']}")
        st.markdown(f"**{meta['artist']}**")
        st.markdown(f"분기: `{meta.get('branch', '?')}`")
        if meta.get("genius_url"):
            st.markdown(f"[Genius 링크]({meta['genius_url']})")
        st.markdown(f"음원: `{meta.get('audio_source', '-')}`")

    with c2:
        st.markdown("**Live features (Librosa)**")
        st.json({k: v for k, v in feats.items() if not k.startswith("_")}, expanded=False)

    st.divider()

    cc1, cc2 = st.columns([2, 1])
    with cc1:
        st.markdown(f"### Sonic texture: `{concept.get('sonic_texture', '-')}`")
        st.markdown(f"### Narrative archetype: `{concept.get('narrative_archetype', '-')}`")

        symbols = concept.get("visual_symbols", [])
        if symbols:
            st.markdown("**Visual symbols**")
            st.markdown(" · ".join(f"**{s}**" for s in symbols))

        if concept.get("reasoning_brief"):
            st.markdown("**Reasoning**")
            st.write(concept["reasoning_brief"])

    with cc2:
        palette = concept.get("color_palette", {}) or {}
        main = palette.get("main", "")
        subs = palette.get("sub", []) or []

        st.markdown("**Color palette**")
        if main:
            st.markdown(
                f"<div style='background:{main};height:64px;border-radius:8px;"
                f"display:flex;align-items:center;justify-content:center;color:white;"
                f"text-shadow:0 0 4px rgba(0,0,0,0.6);font-weight:600;'>"
                f"{main}</div>",
                unsafe_allow_html=True,
            )
        sub_cols = st.columns(max(1, len(subs)))
        for col, hexv in zip(sub_cols, subs):
            with col:
                st.markdown(
                    f"<div style='background:{hexv};height:48px;border-radius:6px;"
                    f"display:flex;align-items:center;justify-content:center;color:white;"
                    f"text-shadow:0 0 4px rgba(0,0,0,0.6);font-size:12px;'>"
                    f"{hexv}</div>",
                    unsafe_allow_html=True,
                )

    with st.expander("전체 JSON"):
        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")


def tab_visual_mapping():
    """Visual Mapping 결과 탭. 미리 생성된 result를 session에서 읽는다."""
    bundle = st.session_state.get("run_bundle", {})
    result = bundle.get("visual")
    if result is None:
        st.info("Visual Mapping이 실행되지 않았습니다. 메인 페이지에서 'Visual Mapping'을 체크하고 실행하세요.")
        return
    _render_visual_concept(result)


# ══════════════════════════════════════════════════════════════
# 페이지 4 — 시각 컨셉 DB 열람
# ══════════════════════════════════════════════════════════════

def tab_visual_db():
    from music_analysis import visual_mapping as vm

    st.caption(f"`{vm.DB_PATH.relative_to(_PROJECT_ROOT)}`에 저장된 캐시 항목.")
    entries = vm.list_cached()
    if not entries:
        st.info("아직 캐시된 항목이 없습니다. Visual Mapping을 한 번 실행하세요.")
        return
    options = [
        f"{e['metadata']['artist']} — {e['metadata']['title']}  ({e['metadata'].get('branch', '?')})"
        for e in entries
    ]
    idx = st.selectbox("항목 선택", range(len(options)), format_func=lambda i: options[i], key="vm_db_select")
    _render_visual_concept(entries[idx])


# ══════════════════════════════════════════════════════════════
# 통합 실행 — 입력 → 선택된 파이프라인 순차 실행 → 결과 번들
# ══════════════════════════════════════════════════════════════

def _run_pipelines(
    *,
    input_path: str,
    audio_bytes: bytes,
    audio_mime: str,
    filename: str,
    title: str,
    artist: str,
    do_analysis: bool,
    do_split: bool,
    do_visual: bool,
    force_visual_refresh: bool,
) -> dict:
    """선택된 파이프라인을 순차 실행하고 결과를 한 dict로 묶어 반환.

    - analysis: pipeline.analyze_track(input_path)
    - split: load_audio_stereo(input_path) → 채널은 즉시, Demucs는 4 stem 모두 미리 분리
    - visual: visual_mapping.process_kpop_visualization(audio_path=input_path)
    """
    bundle: dict = {
        "filename": filename,
        "audio_bytes": audio_bytes,
        "audio_mime": audio_mime,
        "title": title,
        "artist": artist,
    }

    # 1) 음악 분석
    if do_analysis:
        from music_analysis.pipeline import analyze_track
        with st.status("음악 분석 중...", expanded=True) as s:
            inner = st.progress(0.0, text="시작")
            result = analyze_track(input_path, progress_cb=lambda p, m: inner.progress(p, text=m))
            bundle["analysis"] = result
            s.update(label="음악 분석 완료", state="complete")

    # 2) 음원 분리
    if do_split:
        with st.status("음원 분리 (채널 + Demucs 4-stem)...", expanded=True) as s:
            inner = st.progress(0.0, text="오디오 로딩")
            audio, sr = load_audio_stereo(input_path)
            inner.progress(0.1, text="채널 분리는 즉시 가능 / Demucs 분리 시작")
            try:
                stems = separate_stems(audio, sr, progress_cb=lambda p, m: inner.progress(0.1 + 0.9 * p, text=m))
                demucs_sr = stems.pop("_sr")
            except Exception as e:
                st.warning(f"Demucs 분리 실패 — 채널 분리만 사용 가능: {e}")
                stems = None
                demucs_sr = sr
            bundle["splitter"] = {
                "audio": audio,
                "sr": sr,
                "stem": Path(filename).stem,
                "demucs_stems": stems,
                "demucs_sr": demucs_sr,
            }
            s.update(label="음원 분리 완료", state="complete")

    # 3) Visual Mapping
    if do_visual:
        from music_analysis import visual_mapping as vm
        with st.status("Visual Mapping (Genius + Gemini)...", expanded=True) as s:
            inner = st.progress(0.0, text="시작")
            try:
                result = vm.process_kpop_visualization(
                    title.strip(),
                    artist.strip(),
                    audio_path=input_path,
                    force_refresh=force_visual_refresh,
                    progress_cb=lambda p, m: inner.progress(p, text=m),
                )
                bundle["visual"] = result
                s.update(label="Visual Mapping 완료", state="complete")
            except Exception as e:
                s.update(label=f"Visual Mapping 실패: {e}", state="error")

    return bundle


def _input_form():
    """메인 입력 폼. 제출되면 선택된 파이프라인을 즉시 실행한다."""
    from music_analysis import visual_mapping as vm

    st.title("🎧 Spatial Audio Visualization")
    st.caption("K-pop 트랙을 한 번에 분석·분리·시각 매핑.")

    with st.expander("🔑 API 키 상태 (Visual Mapping용)", expanded=False):
        genius_token, gemini_key = vm.load_env()
        st.write({"GENIUS_ACCESS_TOKEN": bool(genius_token), "GEMINI_API_KEY": bool(gemini_key)})
        if not (genius_token and gemini_key):
            st.warning("Visual Mapping을 사용하려면 `.env`에 두 키를 넣어야 합니다. `.env.example` 참고.")

    with st.form("main_input"):
        uploaded = st.file_uploader(
            "음원 파일",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTS],
            help=f"지원 포맷: {', '.join(SUPPORTED_EXTS)}",
        )
        c1, c2 = st.columns(2)
        with c1:
            title = st.text_input("곡 제목", placeholder="예: Supernova")
        with c2:
            artist = st.text_input("아티스트", placeholder="예: aespa")

        st.markdown("**실행할 파이프라인**")
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            do_analysis = st.checkbox("음악 분석", value=True, help="피치 / 구조 경계 / 분위기")
        with cc2:
            do_split = st.checkbox("음원 분리", value=True, help="채널 + Demucs 4-stem")
        with cc3:
            do_visual = st.checkbox("Visual Mapping", value=True, help="Genius + Gemini 분류")
        with cc4:
            force_visual = st.checkbox("Visual 캐시 무시", value=False)

        submitted = st.form_submit_button("실행", type="primary", use_container_width=True)

    if not submitted:
        return False

    # 검증
    errors = []
    if uploaded is None:
        errors.append("음원 파일을 업로드하세요.")
    if not title.strip() or not artist.strip():
        errors.append("곡 제목과 아티스트는 필수입니다.")
    if not (do_analysis or do_split or do_visual):
        errors.append("최소 한 개의 파이프라인을 선택하세요.")
    if errors:
        for e in errors:
            st.error(e)
        return False

    input_path = save_uploaded(uploaded)
    audio_bytes = uploaded.getvalue() if hasattr(uploaded, "getvalue") else Path(input_path).read_bytes()
    audio_mime = uploaded.type or "audio/wav"
    filename = uploaded.name

    try:
        bundle = _run_pipelines(
            input_path=input_path,
            audio_bytes=audio_bytes,
            audio_mime=audio_mime,
            filename=filename,
            title=title.strip(),
            artist=artist.strip(),
            do_analysis=do_analysis,
            do_split=do_split,
            do_visual=do_visual,
            force_visual_refresh=force_visual,
        )
    finally:
        try:
            os.unlink(input_path)
        except OSError:
            pass

    st.session_state["run_bundle"] = bundle
    # 새 ZIP 캐시 초기화 (이전 분석의 export ZIP이 남아있을 수 있음)
    st.session_state.pop("analysis_zip", None)
    st.session_state.pop("analysis_zip_name", None)
    return True


def main():
    st.set_page_config(
        page_title="Spatial Audio Visualization",
        layout="wide",
    )

    just_ran = _input_form()

    bundle = st.session_state.get("run_bundle")
    if bundle is None:
        st.info("위 폼을 채우고 '실행'을 눌러 분석을 시작하세요.")
        return

    if just_ran:
        st.success(f"✅ '{bundle['title']} — {bundle['artist']}' 처리 완료. 아래 탭에서 결과를 확인하세요.")

    st.divider()

    tab_analysis, tab_split, tab_visual, tab_db = st.tabs([
        "음악 분석",
        "음원 분리",
        "Visual Mapping",
        "캐시 DB",
    ])
    with tab_analysis:
        tab_music_analysis()
    with tab_split:
        tab_audio_splitter()
    with tab_visual:
        tab_visual_mapping()
    with tab_db:
        tab_visual_db()


if __name__ == "__main__":
    main()
else:
    main()
