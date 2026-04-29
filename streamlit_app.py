"""Spatial Audio Visualization — 통합 Streamlit 앱.

사이드바에서 4가지 도구를 전환:
  1. 음원 분리         — 채널/트랙 단위로 음원을 쪼갠다 (Demucs)
  2. 음악 분석          — 피치 / 구조 경계 / 분위기 (Librosa)
  3. K-pop Visual Mapping — Genius + Gemini로 비주얼 컨셉 JSON 생성
  4. 시각 컨셉 DB       — 캐시된 비주얼 매핑 결과 열람

실행: streamlit run streamlit_app.py
"""

from __future__ import annotations

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


def page_audio_splitter():
    st.title("🎧 음원 분리")
    st.caption("음원 파일을 채널(L/R/Mid/Side) 또는 트랙(보컬/멜로디/비트) 단위로 분리합니다.")

    uploaded = st.file_uploader(
        "음원 파일 업로드",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTS],
        key="splitter_uploader",
        help=f"지원 포맷: {', '.join(SUPPORTED_EXTS)}",
    )
    if uploaded is None:
        st.info("음원 파일을 업로드하세요.")
        return

    input_path = save_uploaded(uploaded)
    stem_name = Path(uploaded.name).stem

    with st.spinner("오디오 로딩 중..."):
        try:
            audio, sr = load_audio_stereo(input_path)
        except Exception as e:
            st.error(f"오디오 로딩 실패: {e}")
            return

    duration = audio.shape[1] / sr
    c1, c2, c3 = st.columns(3)
    c1.metric("길이", f"{duration:.1f} s")
    c2.metric("샘플레이트", f"{sr} Hz")
    c3.metric("채널", audio.shape[0])

    st.audio(uploaded.getvalue() if hasattr(uploaded, "getvalue") else input_path)
    st.divider()

    mode = st.radio(
        "분리 방식",
        ["채널 분리 (L / R / Mid / Side)", "트랙 분리 (보컬 / 멜로디 / 비트)"],
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
        if st.button("분리 실행", type="primary", disabled=len(channel_opts) == 0, key="splitter_run_ch"):
            splits = split_channels(audio)
            st.session_state["splitter_results"] = {k: splits[k] for k in channel_opts}
            st.session_state["splitter_results_sr"] = sr
            st.session_state["splitter_stem"] = stem_name
            st.session_state["splitter_mode_key"] = "channel"

    else:
        st.subheader("트랙 분리")
        st.caption("Meta Demucs v4 (htdemucs). 첫 실행 시 모델 다운로드(~80MB).")

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

        if st.button("분리 실행", type="primary", key="splitter_run_track"):
            prog = st.progress(0.0, text="시작")
            try:
                with st.spinner("트랙 분리 중... (곡 길이에 따라 수십 초~수 분)"):
                    stems = separate_stems(audio, sr, progress_cb=lambda p, m: prog.progress(p, text=m))
            except Exception as e:
                st.error(f"분리 실패: {e}")
                return
            out_sr = stems.pop("_sr")
            st.session_state["splitter_results"] = apply_preset(stems, preset)
            st.session_state["splitter_results_sr"] = out_sr
            st.session_state["splitter_stem"] = stem_name
            st.session_state["splitter_mode_key"] = "track"
            prog.empty()

    if "splitter_results" in st.session_state:
        st.divider()
        st.subheader("결과")
        results = st.session_state["splitter_results"]
        out_sr = st.session_state["splitter_results_sr"]
        stem = st.session_state["splitter_stem"]
        mode_key = st.session_state["splitter_mode_key"]

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
            audio_out = wav_bytes(data, out_sr)
            col1, col2 = st.columns([4, 1])
            with col1:
                st.audio(audio_out, format="audio/wav")
            with col2:
                st.download_button(
                    "⬇ WAV",
                    data=audio_out,
                    file_name=f"{stem}_{key}.wav",
                    mime="audio/wav",
                    key=f"dl_split_{key}",
                )

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

    try:
        os.unlink(input_path)
    except OSError:
        pass


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


def page_music_analysis():
    from music_analysis.pipeline import analyze_track, export_result

    st.title("🎼 음악 분석 파이프라인")
    st.caption("피치(pYIN) · 구조 경계(라플라시안 분해) · 분위기(valence/energy/tension)를 한번에 추출합니다.")

    uploaded = st.file_uploader(
        "음원 파일 업로드 (.wav / .flac / .mp3)",
        type=["wav", "flac", "mp3"],
        key="analysis_uploader",
    )
    if uploaded is None:
        st.info("분석할 음원을 업로드하세요.")
        return

    input_path = save_uploaded(uploaded)
    st.audio(uploaded.getvalue() if hasattr(uploaded, "getvalue") else input_path)

    if st.button("분석 실행", type="primary", key="analysis_run"):
        prog = st.progress(0.0, text="시작")
        try:
            with st.spinner("분석 중... (곡 길이에 따라 수십 초~수 분)"):
                result = analyze_track(input_path, progress_cb=lambda p, m: prog.progress(p, text=m))
        except Exception as e:
            st.error(f"분석 실패: {e}")
            return
        prog.empty()
        # session에 저장 (NumPy 큰 배열 포함되므로 그대로 보관)
        st.session_state["analysis_result"] = result
        st.session_state["analysis_filename"] = uploaded.name

    result = st.session_state.get("analysis_result")
    if result is None:
        return

    st.divider()
    st.subheader("개요")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("길이", f"{result['duration_sec']:.1f} s")
    c2.metric("Tempo", f"{result['tempo_bpm']:.1f} BPM")
    ps = result["pitch"]["summary"]
    if ps:
        c3.metric("피치 중앙값", f"{ps['f0_median_hz']:.0f} Hz")
        c4.metric("Voiced 비율", f"{ps['voiced_ratio']*100:.0f} %")

    st.subheader("플롯")
    try:
        fig = _plot_analysis(result)
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.warning(f"플롯 생성 실패: {e}")

    st.subheader("섹션별 분위기")
    sections = result["mood"]["sections"]
    if sections:
        st.dataframe(
            sections,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("섹션이 감지되지 않았습니다.")

    st.subheader("경계")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**구조 경계 (초)**")
        st.write([round(b, 2) for b in result["segmentation"]["boundary_times"]])
    with c2:
        st.markdown("**분위기 전환부 (초)**")
        st.write([round(b, 2) for b in result["mood"]["mood_boundaries"]["times"]])

    st.divider()
    st.subheader("내보내기")
    st.caption("L/R wav + sections/pitch/mood_frames/boundaries/meta CSV를 ZIP으로 묶어 다운로드.")

    if st.button("ZIP 생성", key="analysis_export"):
        with st.spinner("CSV/WAV 생성 중..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                # export_result는 result["path"]의 stem을 파일명으로 사용
                # 업로드 파일명으로 보이도록 path 임시 교체
                orig_path = result["path"]
                pretty_stem = Path(st.session_state.get("analysis_filename", orig_path)).stem
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

    try:
        os.unlink(input_path)
    except OSError:
        pass


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


def page_visual_mapping():
    from music_analysis import visual_mapping as vm

    st.title("🌌 K-pop Visual Mapping")
    st.caption("iTunes/YouTube에서 음원 30초 → Librosa feature → Genius 가사(있으면) → Gemini 분류 JSON.")

    genius_token, gemini_key = vm.load_env()
    with st.expander("API 키 상태", expanded=not (genius_token and gemini_key)):
        st.write({"GENIUS_ACCESS_TOKEN": bool(genius_token), "GEMINI_API_KEY": bool(gemini_key)})
        if not (genius_token and gemini_key):
            st.warning("프로젝트 루트의 `.env`에 키를 넣어주세요. `.env.example` 참고.")

    with st.form("vm_form"):
        c1, c2 = st.columns(2)
        with c1:
            title = st.text_input("곡 제목", placeholder="예: Supernova")
        with c2:
            artist = st.text_input("아티스트", placeholder="예: aespa")
        youtube_url = st.text_input(
            "YouTube URL (선택 — iTunes에 없거나 신곡일 때)",
            placeholder="https://youtu.be/...",
        )
        c3, c4 = st.columns([1, 1])
        with c3:
            force = st.checkbox("캐시 무시하고 새로 처리", value=False)
        with c4:
            submitted = st.form_submit_button("실행", type="primary", use_container_width=True)

    if submitted:
        if not title.strip() or not artist.strip():
            st.error("곡 제목과 아티스트는 필수입니다.")
        else:
            prog = st.progress(0.0, text="시작")
            try:
                with st.spinner("처리 중..."):
                    result = vm.process_kpop_visualization(
                        title.strip(),
                        artist.strip(),
                        youtube_url=youtube_url.strip() or None,
                        force_refresh=force,
                        progress_cb=lambda p, m: prog.progress(p, text=m),
                    )
                st.session_state["vm_last_result"] = result
                st.success("완료. 결과는 `music_analysis/kpop_visual_db.json`에 캐시됩니다.")
            except Exception as e:
                st.error(f"실패: {e}")
            finally:
                prog.empty()

    result = st.session_state.get("vm_last_result")
    if result is not None:
        st.divider()
        _render_visual_concept(result)


# ══════════════════════════════════════════════════════════════
# 페이지 4 — 시각 컨셉 DB 열람
# ══════════════════════════════════════════════════════════════

def page_visual_db():
    from music_analysis import visual_mapping as vm

    st.title("📚 시각 컨셉 DB")
    st.caption(f"`{vm.DB_PATH.relative_to(_PROJECT_ROOT)}`에 저장된 캐시 항목.")

    entries = vm.list_cached()
    if not entries:
        st.info("아직 캐시된 항목이 없습니다. K-pop Visual Mapping에서 곡을 처리하세요.")
        return

    options = [
        f"{e['metadata']['artist']} — {e['metadata']['title']}  ({e['metadata'].get('branch', '?')})"
        for e in entries
    ]
    idx = st.selectbox("항목 선택", range(len(options)), format_func=lambda i: options[i])
    _render_visual_concept(entries[idx])


# ══════════════════════════════════════════════════════════════
# 메인 — 사이드바 네비게이션
# ══════════════════════════════════════════════════════════════

PAGES = {
    "🎧 음원 분리": page_audio_splitter,
    "🎼 음악 분석": page_music_analysis,
    "🌌 K-pop Visual Mapping": page_visual_mapping,
    "📚 시각 컨셉 DB": page_visual_db,
}


def main():
    st.set_page_config(
        page_title="Spatial Audio Visualization",
        page_icon="🎧",
        layout="wide",
    )

    with st.sidebar:
        st.markdown("## Spatial Audio")
        st.caption("음원을 분리·분석하고 비주얼 컨셉으로 매핑합니다.")
        st.divider()
        choice = st.radio(
            "도구",
            list(PAGES.keys()),
            key="page_choice",
            label_visibility="collapsed",
        )
        st.divider()
        with st.expander("이 도구는?"):
            st.markdown(
                "- **음원 분리** — 채널(L/R/Mid/Side) 또는 트랙(보컬/멜로디/베이스/비트)으로 쪼갭니다.\n"
                "- **음악 분석** — 피치 궤적, 구조 경계, 섹션별 분위기를 추출.\n"
                "- **K-pop Visual Mapping** — Genius+Gemini로 비주얼 컨셉 JSON 생성.\n"
                "- **시각 컨셉 DB** — 캐시된 매핑 결과 열람.\n"
            )

    PAGES[choice]()


if __name__ == "__main__":
    main()
else:
    main()
