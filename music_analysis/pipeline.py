"""Music analysis pipeline — pitch / segmentation / mood.

`music_analysis_pipeline.ipynb`의 함수들을 그대로 모듈화한 것. 노트북에서
검증된 로직이므로 시그니처/동작은 유지하고 import 가능한 형태로만 재정리.

기본 사용:

    from music_analysis.pipeline import analyze_track
    result = analyze_track("samples/track.mp3")
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List

import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from sklearn.cluster import KMeans

SUPPORTED_EXT = {".wav", ".flac", ".mp3", ".m4a", ".mp4", ".aac", ".ogg"}
SR = 22050


# ──────────────────────────────────────────────────────────────
# 1. 로딩
# ──────────────────────────────────────────────────────────────

def load_audio(path: str, sr: int = SR) -> Dict:
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported format: {ext}. Use one of {SUPPORTED_EXT}")
    y_stereo, sr_out = librosa.load(path, sr=sr, mono=False)
    if y_stereo.size == 0:
        raise ValueError(f"Empty audio: {path}")
    if y_stereo.ndim == 1:
        y_stereo = np.stack([y_stereo, y_stereo])
    return {
        "left": y_stereo[0],
        "right": y_stereo[1],
        "mono": (y_stereo[0] + y_stereo[1]) / 2.0,
        "sr": sr_out,
    }


# ──────────────────────────────────────────────────────────────
# 2. 피치
# ──────────────────────────────────────────────────────────────

def extract_pitch(y: np.ndarray, sr: int) -> Dict:
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=2048,
    )
    times = librosa.times_like(f0, sr=sr)

    with np.errstate(invalid="ignore", divide="ignore"):
        midi = librosa.hz_to_midi(f0)

    valid = ~np.isnan(f0)
    summary = {}
    if valid.any():
        summary = {
            "f0_median_hz": float(np.nanmedian(f0)),
            "midi_median": float(np.nanmedian(midi)),
            "midi_range": float(np.nanmax(midi) - np.nanmin(midi)),
            "voiced_ratio": float(valid.mean()),
        }

    return {
        "times": times,
        "f0_hz": f0,
        "midi": midi,
        "voiced_prob": voiced_prob,
        "summary": summary,
    }


# ──────────────────────────────────────────────────────────────
# 3. 구조 경계 (라플라시안 스펙트럴 분해)
# ──────────────────────────────────────────────────────────────

def segment_song(y: np.ndarray, sr: int, min_segment_sec: float = 8.0) -> Dict:
    duration = librosa.get_duration(y=y, sr=sr)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    tempo_val = float(np.atleast_1d(tempo)[0])

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    if len(beats) < 8:
        beats = np.arange(0, chroma.shape[1], 8)

    Csync = librosa.util.sync(chroma, beats, aggregate=np.median)
    Msync = librosa.util.sync(mfcc, beats, aggregate=np.mean)

    n = Csync.shape[1]
    if n < 4:
        return {
            "tempo": tempo_val,
            "boundary_times": [0.0, float(duration)],
            "labels": [0],
        }

    R = librosa.segment.recurrence_matrix(
        Csync, width=3, mode="affinity", sym=True,
    )
    df = librosa.segment.timelag_filter(median_filter)
    R = df(R, size=(1, 7))

    path_distance = np.sum(np.diff(Msync, axis=1) ** 2, axis=0)
    sigma = np.median(path_distance) + 1e-9
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(R, axis=1)
    mu = deg_path.dot(deg_path + deg_rec) / (np.sum((deg_path + deg_rec) ** 2) + 1e-9)
    A = mu * R + (1 - mu) * R_path

    L, _ = csgraph_laplacian(A, normed=True, return_diag=True)
    try:
        evals, evecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(L + 1e-6 * np.eye(L.shape[0]))

    k = int(np.clip(round(duration / 25.0), 3, 10))
    k = min(k, n - 1)

    X = evecs[:, :k]
    X = librosa.util.normalize(X, norm=2, axis=1)
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels_beat = km.fit_predict(X)
    labels_beat = median_filter(labels_beat, size=9, mode="nearest")

    beat_times = librosa.frames_to_time(beats, sr=sr)
    beat_times = np.concatenate([beat_times, [duration]])

    change = np.flatnonzero(np.diff(labels_beat)) + 1
    boundary_beats = np.unique(np.concatenate([[0], change, [n]])).tolist()

    def _beat_time(bi):
        return float(beat_times[min(bi, len(beat_times) - 1)])

    def _seg_label(i):
        return int(labels_beat[min(boundary_beats[i], n - 1)])

    changed = True
    while changed and len(boundary_beats) > 2:
        changed = False
        for i in range(len(boundary_beats) - 1):
            start_t = _beat_time(boundary_beats[i])
            end_t = _beat_time(boundary_beats[i + 1])
            if end_t - start_t >= min_segment_sec:
                continue
            cur = _seg_label(i)
            prev_lbl = _seg_label(i - 1) if i > 0 else None
            next_lbl = _seg_label(i + 1) if i + 1 < len(boundary_beats) - 1 else None
            if prev_lbl is None and next_lbl is None:
                break
            if prev_lbl == cur:
                boundary_beats.pop(i)
            elif next_lbl == cur:
                boundary_beats.pop(i + 1)
            elif prev_lbl is None:
                boundary_beats.pop(i + 1)
            elif next_lbl is None:
                boundary_beats.pop(i)
            else:
                prev_len = start_t - _beat_time(boundary_beats[i - 1])
                next_len = _beat_time(boundary_beats[i + 2]) - end_t
                if prev_len >= next_len:
                    boundary_beats.pop(i)
                else:
                    boundary_beats.pop(i + 1)
            changed = True
            break

    boundary_times = [_beat_time(bi) for bi in boundary_beats]
    boundary_times = sorted(set(round(t, 3) for t in boundary_times))

    seg_labels = []
    for i in range(len(boundary_beats) - 1):
        start_b = boundary_beats[i]
        seg_labels.append(int(labels_beat[min(start_b, n - 1)]))

    return {
        "tempo": tempo_val,
        "boundary_times": boundary_times,
        "labels": seg_labels,
    }


# ──────────────────────────────────────────────────────────────
# 4. 분위기
# ──────────────────────────────────────────────────────────────

def _zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x) + 1e-9
    return (x - mu) / sd


_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _estimate_mode(chroma: np.ndarray) -> float:
    mean_chroma = chroma.mean(axis=1)
    if mean_chroma.sum() < 1e-6:
        return 0.0
    best_major = max(np.corrcoef(np.roll(mean_chroma, -i), _MAJOR_PROFILE)[0, 1] for i in range(12))
    best_minor = max(np.corrcoef(np.roll(mean_chroma, -i), _MINOR_PROFILE)[0, 1] for i in range(12))
    return float(best_major - best_minor)


def _label_mood(valence: float, energy: float, tension: float) -> str:
    v = "bright" if valence > 0.2 else ("dark" if valence < -0.2 else "neutral")
    e = "high-energy" if energy > 0.3 else ("calm" if energy < -0.3 else "moderate")
    parts = [e, v]
    if tension > 0.5:
        parts.append("tense")
    elif tension < -0.5:
        parts.append("relaxed")
    return " / ".join(parts)


def compute_mood_frames(y: np.ndarray, sr: int, hop: int = 512) -> Dict:
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=sr)

    N = len(rms)
    onset_env = onset_env[:N]
    chroma = chroma[:, :N]

    z_rms = _zscore(rms)
    z_centroid = _zscore(centroid[:N])
    z_onset = _zscore(onset_env)
    z_flatness = _zscore(flatness[:N])
    z_tonnetz = _zscore(tonnetz[:, :N].std(axis=0))

    win = max(1, int(sr / hop * 2))
    mode_trace = np.zeros(N)
    for i in range(N):
        lo = max(0, i - win)
        hi = min(N, i + win)
        mode_trace[i] = _estimate_mode(chroma[:, lo:hi])

    energy = 0.6 * z_rms + 0.4 * z_onset
    brightness = z_centroid
    valence = 0.7 * mode_trace + 0.3 * np.tanh(brightness)
    tension = 0.5 * z_tonnetz + 0.5 * z_flatness

    frame_times = librosa.frames_to_time(np.arange(N), sr=sr, hop_length=hop)
    return {
        "times": frame_times,
        "valence": valence,
        "energy": energy,
        "tension": tension,
        "brightness": brightness,
        "chroma": chroma,
        "hop": hop,
    }


def summarize_sections(mood_frames: Dict, boundary_times: List[float]) -> List[Dict]:
    times = mood_frames["times"]
    results = []
    for idx, (start, end) in enumerate(zip(boundary_times[:-1], boundary_times[1:])):
        mask = (times >= start) & (times < end)
        if not mask.any():
            continue
        v = float(mood_frames["valence"][mask].mean())
        e = float(mood_frames["energy"][mask].mean())
        t = float(mood_frames["tension"][mask].mean())
        results.append({
            "section_idx": idx,
            "start": float(start),
            "end": float(end),
            "duration": round(end - start, 2),
            "valence": round(v, 3),
            "energy": round(e, 3),
            "tension": round(t, 3),
            "mood_label": _label_mood(v, e, t),
        })
    return results


def mood_novelty_boundaries(mood_frames: Dict, window_sec: float = 4.0, min_gap_sec: float = 6.0) -> Dict:
    times = mood_frames["times"]
    if len(times) < 4:
        return {"times": [], "novelty": np.zeros_like(times), "frame_times": times}

    V = np.stack([mood_frames["valence"], mood_frames["energy"], mood_frames["tension"]], axis=0)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.05
    w = max(4, int(round(window_sec / dt)))
    min_distance = max(1, int(round(min_gap_sec / dt)))

    N = V.shape[1]
    novelty = np.zeros(N)
    for i in range(w, N - w):
        left = V[:, i - w : i].mean(axis=1)
        right = V[:, i : i + w].mean(axis=1)
        novelty[i] = np.linalg.norm(right - left)

    if novelty.max() > 0:
        novelty = novelty / novelty.max()
    peaks, _ = find_peaks(novelty, distance=min_distance, prominence=0.2)
    return {
        "times": [float(times[p]) for p in peaks],
        "novelty": novelty,
        "frame_times": times,
    }


def analyze_mood(y: np.ndarray, sr: int, boundary_times: List[float]) -> Dict:
    frames = compute_mood_frames(y, sr)
    sections = summarize_sections(frames, boundary_times)
    novelty = mood_novelty_boundaries(frames)
    return {
        "frames": frames,
        "sections": sections,
        "mood_boundaries": novelty,
    }


# ──────────────────────────────────────────────────────────────
# 5. 파이프라인
# ──────────────────────────────────────────────────────────────

def analyze_track(path: str, progress_cb=None) -> Dict:
    """Pitch + segmentation + mood. progress_cb(pct, msg) optional."""
    if progress_cb:
        progress_cb(0.05, "오디오 로딩")
    audio = load_audio(path)
    y_mono = audio["mono"]
    sr = audio["sr"]
    duration = librosa.get_duration(y=y_mono, sr=sr)

    if progress_cb:
        progress_cb(0.2, "피치(pYIN) 추출")
    pitch = extract_pitch(y_mono, sr)

    if progress_cb:
        progress_cb(0.5, "구조 경계 감지")
    seg = segment_song(y_mono, sr)

    if progress_cb:
        progress_cb(0.8, "분위기 분석")
    mood = analyze_mood(y_mono, sr, seg["boundary_times"])

    if progress_cb:
        progress_cb(1.0, "완료")

    return {
        "path": path,
        "duration_sec": round(float(duration), 2),
        "sr": sr,
        "tempo_bpm": round(seg["tempo"], 2),
        "pitch": pitch,
        "segmentation": seg,
        "mood": mood,
        "_audio": {
            "left": audio["left"],
            "right": audio["right"],
            "mono": y_mono,
            "sr": sr,
        },
    }


# ──────────────────────────────────────────────────────────────
# 6. CSV / WAV 내보내기
# ──────────────────────────────────────────────────────────────

def export_result(result: Dict, out_dir: str | None = None) -> str:
    """L/R wav + 분석 CSV들을 디스크에 저장. 출력 디렉토리 경로 반환."""
    path = result["path"]
    stem = os.path.splitext(os.path.basename(path))[0]
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(path), f"{stem}_analysis")
    os.makedirs(out_dir, exist_ok=True)

    sr = result["_audio"]["sr"]

    sf.write(os.path.join(out_dir, f"{stem}_L.wav"), result["_audio"]["left"], sr)
    sf.write(os.path.join(out_dir, f"{stem}_R.wav"), result["_audio"]["right"], sr)

    sections = result["mood"]["sections"]
    with open(os.path.join(out_dir, f"{stem}_sections.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "section_idx", "start", "end", "duration",
            "valence", "energy", "tension", "mood_label",
        ])
        w.writeheader()
        w.writerows(sections)

    pitch = result["pitch"]
    with open(os.path.join(out_dir, f"{stem}_pitch.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "f0_hz", "midi", "voiced_prob"])
        for t, hz, m, vp in zip(pitch["times"], pitch["f0_hz"], pitch["midi"], pitch["voiced_prob"]):
            w.writerow([
                round(t, 4),
                round(hz, 3) if not np.isnan(hz) else "",
                round(m, 3) if not np.isnan(m) else "",
                round(vp, 4),
            ])

    mf = result["mood"]["frames"]
    with open(os.path.join(out_dir, f"{stem}_mood_frames.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "valence", "energy", "tension", "brightness"])
        for i in range(len(mf["times"])):
            w.writerow([
                round(mf["times"][i], 4),
                round(float(mf["valence"][i]), 4),
                round(float(mf["energy"][i]), 4),
                round(float(mf["tension"][i]), 4),
                round(float(mf["brightness"][i]), 4),
            ])

    struct_set = set(result["segmentation"]["boundary_times"])
    mood_set = set(result["mood"]["mood_boundaries"]["times"])
    all_bounds = sorted(struct_set | mood_set)
    with open(os.path.join(out_dir, f"{stem}_boundaries.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "type"])
        for t in all_bounds:
            types = []
            if t in struct_set:
                types.append("structural")
            if t in mood_set:
                types.append("mood")
            w.writerow([round(t, 3), "+".join(types)])

    with open(os.path.join(out_dir, f"{stem}_meta.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerow(["file", path])
        w.writerow(["duration_sec", result["duration_sec"]])
        w.writerow(["sr", sr])
        w.writerow(["tempo_bpm", result["tempo_bpm"]])
        ps = result["pitch"]["summary"]
        for k, v in ps.items():
            w.writerow([k, round(v, 4)])

    return out_dir
