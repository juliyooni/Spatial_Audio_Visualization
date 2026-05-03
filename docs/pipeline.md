# 음원 처리 파이프라인 — 기술 문서

이 문서는 K-pop 곡 한 개를 입력받아 음원 분리 / 음악 분석 / 비주얼 매핑까지 수행하는 전체 파이프라인을 항목별로 정리한다. Streamlit 앱(`streamlit_app.py`)이 입력과 라우팅을 담당하고, 실제 알고리즘은 `music_analysis/pipeline.py`(분석)과 `music_analysis/visual_mapping.py`(매핑) 두 모듈에 들어 있다.

## 0. 한눈에 보기

```
[ 입력 폼 ]
음원 파일 + 곡 제목 + 아티스트 + 실행할 파이프라인 체크박스
        │
        ▼
[ _run_pipelines() ] — 선택된 파이프라인을 순차 실행
        │
        ├── 🎼 음악 분석   → analyze_track()       → pitch / segmentation / mood
        ├── 🎧 음원 분리   → split_channels() + Demucs htdemucs (4-stem)
        └── 🌌 Visual Mapping → process_kpop_visualization()
                                  → Genius (text 분기) 또는 Gemini audio (new 분기)
        │
        ▼
[ st.session_state["run_bundle"] ] — 결과 번들
        │
        ▼
[ 결과 탭 4개 ] 음악 분석 / 음원 분리 / Visual Mapping / 캐시 DB
```

각 파이프라인은 독립적이라 체크박스로 끄고 켤 수 있다. **모두 같은 업로드 파일**을 입력으로 받기 때문에 분석 결과가 서로 일관된다.

---

## 1. 입력과 음원 확보

### 1.1 폼 항목 (`_input_form` in `streamlit_app.py`)

| 항목 | 필수 | 비고 |
|---|---|---|
| 음원 파일 업로드 | ✅ | `.wav .flac .mp3 .mp4 .m4a .aac .ogg` |
| 곡 제목 | ✅ | Visual Mapping 캐시 키 + Genius 조회 |
| 아티스트 | ✅ | 같은 용도 |
| 🎼 음악 분석 체크박스 | — | 기본 ON |
| 🎧 음원 분리 체크박스 | — | 기본 ON |
| 🌌 Visual Mapping 체크박스 | — | 기본 ON |
| Visual 캐시 무시 체크박스 | — | 기본 OFF — 켜면 캐시 히트 무시하고 Gemini 재호출 |

폼 제출 시 `save_uploaded()`가 업로드 바이트를 확장자 보존한 임시 파일에 쓴 뒤 그 경로를 모든 파이프라인에 흘려보낸다. 임시 파일은 `_run_pipelines()` 종료 후 `os.unlink`로 정리.

### 1.2 YouTube 입력은 왜 없는가

이전 버전엔 "파일이 없으면 yt-dlp로 풀 트랙 다운로드" 경로가 있었는데, **YouTube가 클라우드 IP에서 오는 yt-dlp 요청을 거의 봇으로 간주**해서 Streamlit Cloud 같은 호스팅 환경에선 첫 시도부터 HTTP 403이 반복된다. player_client 체인 + 브라우저 쿠키 fallback까지 시도했지만 클라우드에선 브라우저 쿠키 자체가 존재하지 않아 의미가 없어, 입력란을 제거하고 사용자가 본인 PC에서 받아 업로드하는 흐름으로 단순화했다.

`download_youtube_audio()` 함수 자체는 `visual_mapping.py`에 그대로 남아있어 노트북에서 ad-hoc하게 호출 가능. **로컬 머신에서 실행할 때만** 안정적으로 동작한다.

### 1.3 사용 외부 라이브러리

| 라이브러리 | 용도 |
|---|---|
| `streamlit` | UI / 폼 / 세션 상태 |
| `streamlit.components.v1` | WaveSurfer.js HTML 임베드 |
| `soundfile` | 업로드 파일 디코딩 / WAV 직렬화 |
| `librosa` | 디코딩 fallback (ffmpeg/audioread 경유), 리샘플링 |

---

## 2. 음원 분리 (음원 분리 탭)

### 2.1 채널 분리

`split_channels(data)`. (channels, samples)을 받아 4개의 stereo (2, samples) 출력을 만든다.

| 출력 | 정의 | 출력 레이아웃 | 의도 |
|---|---|---|---|
| **Left** | `data[0]` | `[L, 0]` | **왼쪽 스피커에서만 들림.** 오른쪽엔 무음. |
| **Right** | `data[1]` | `[0, R]` | **오른쪽 스피커에서만 들림.** 왼쪽엔 무음. |
| **Mid** | `(L + R) / 2` | `[M, M]` | 가운데에 정위된 성분 (보컬 / 베이스). 양쪽 모두에서 들림. |
| **Side** | `(L - R) / 2` | `[S, S]` | 좌우 차이만 남은 성분 (스테레오 와이드 신스 / 리버브 꼬리). 양쪽 모두에서 들림. |

L/R은 모니터링/스피커 라우팅에서 **물리적으로 한쪽 채널에서만 재생**되도록 반대쪽을 0으로 채운다. 헤드폰으로 들으면 한쪽 귀에서만 소리가 나야 정상.

Mid/Side는 "왼쪽 채널 / 오른쪽 채널"이라는 개념 자체가 없는 신호(L+R 합 / L−R 차)이므로 **듀얼 모노**로 양쪽에 같은 파형을 보낸다.

모노 입력은 left를 양쪽에 복제한 상태로 들어오므로 right = left, mid = L, side = 0이 된다.

즉시 계산 — Demucs와 달리 모델 로딩이 없다.

### 2.2 Demucs 4-stem 분리

`separate_stems(data, sr)`. Meta의 **htdemucs** (Hybrid Transformer Demucs v4) 사전학습 모델 사용.

#### 디바이스 선택 (`load_demucs_model`)

```
CUDA 사용 가능 → cuda
elif MPS 사용 가능 → mps    (Apple Silicon)
else → cpu
```

`@st.cache_resource`로 캐시되어 같은 세션에선 모델을 한 번만 로드.

#### 분리 절차

1. 입력을 모델 샘플레이트(`model.samplerate`, htdemucs는 44100)로 librosa 리샘플
2. `(wav - mean) / std`로 z-score 정규화 (Demucs의 학습 가정과 일치)
3. `apply_model(model, wav, shifts=1, overlap=0.25)` — `shifts=1`은 시간축으로 약간 밀어 평균내는 augmentation, `overlap=0.25`는 25% 오버랩 윈도우로 chunked 처리
4. 출력에 `* std + mean`으로 정규화 역연산
5. 4개 stem(`vocals`, `drums`, `bass`, `other`) + `_sr` 키를 dict로 반환

5분 K-pop 한 곡 기준 CPU에서 30~90초, MPS에서 10~30초 정도 걸림 (Apple Silicon 기준 경험치).

### 2.3 프리셋 합성 (`apply_preset`)

Demucs 4-stem은 항상 다 분리한 뒤 결과 탭에서 **프리셋만 바꾸면 즉시 재합성**된다. 합성은 단순 합산 — `sum(stems[s] for s in stem_list)`. 13개 프리셋 정의 ([streamlit_app.py:159-182](../streamlit_app.py#L159-L182)):

| 프리셋 | 출력 트랙 | 합성 식 |
|---|---|---|
| 3트랙 (기본) | `vocals` / `melody` / `drums` | melody = bass + other |
| 4트랙 | `vocals` / `melody` / `bass` / `drums` | 분리된 그대로 |
| 보컬만 | `vocals` | — |
| 멜로디만 | `melody = other` | — |
| 베이스만 / 비트만 | 단일 stem | — |
| 반주만 | `instrumental = drums + bass + other` | 보컬 제외 |
| 2트랙 조합 6종 | 보컬+멜로디 / 보컬+비트 / … | 두 stem 합산 |

프리셋 변경에 모델 재추론이 필요 없어 UI 응답이 즉시. 결과는 개별 WAV 다운로드 + 전체 ZIP 다운로드 버튼으로 내보낼 수 있다.

---

## 3. 음악 분석 파이프라인 (음악 분석 탭)

`analyze_track(path)`이 부르는 3단계 — pitch / segmentation / mood. 모두 `librosa.load(sr=22050, mono=True)`로 리샘플 후 모노 다운믹스한 신호를 입력으로 쓴다 (정확도보다 일관성 우선).

### 3.1 피치 (`extract_pitch`)

- **알고리즘**: `librosa.pyin` (probabilistic YIN) — frame_length=2048, fmin=C2 (~65 Hz), fmax=C7 (~2093 Hz)
- 무성 구간의 `f0`는 NaN으로 보존 (보간하지 않음)
- Hz → MIDI semitone 변환은 같은 NaN 마스크를 유지
- **요약값** (frame 단위 추적과 별도): `f0_median_hz`, `midi_median`, `midi_range`, `voiced_ratio`

해석 시 주의: pYIN은 단성(monophonic) 가정이라 다성악 믹스에선 "지배적인 음고"의 근사로만 본다. K-pop 풀 믹스에서는 보컬 라인이 잡히지만 합창/하모니 구간에서는 가장 강한 한 음만 따라간다.

### 3.2 구조 경계 — 라플라시안 스펙트럴 분해

`segment_song(y, sr, min_segment_sec=8.0)` — McFee & Ellis (2014).

#### 알고리즘

1. **비트 동기 특징** — `librosa.beat.beat_track`으로 비트를 잡고, chroma(화성)와 MFCC(음색) 13차를 비트 단위로 묶는다(`librosa.util.sync`). 이후 모든 거리 계산이 프레임이 아니라 비트 기준이라 텀포에 의존하지 않는다.
2. **재발 그래프(R)** — chroma 비트 시퀀스에서 `librosa.segment.recurrence_matrix(width=3, mode="affinity")`로 만든 affinity 행렬. 이게 "곡 멀리 떨어진 두 비트가 화성적으로 얼마나 비슷한가"를 잡는다. 라벨 깜빡임을 줄이려고 `timelag_filter(median_filter, size=(1,7))`로 대각선 방향 평활.
3. **국소 유사도(R_path)** — 인접 비트 간 MFCC L2 거리에 `exp(-d/σ)`를 씌워 인접한 두 비트가 음색적으로 얼마나 이어지는지 표현. 곡의 흐름을 끊지 않게 하는 역할.
4. **그래프 결합** — McFee의 balancing μ로 `A = μ·R + (1-μ)·R_path`. 두 그래프의 degree 벡터로 계산해 둘의 영향력을 자동 균형.
5. **정규화 라플라시안 → 고유분해 → KMeans** — `csgraph_laplacian(A, normed=True)`의 첫 k개 고유벡터를 행 단위 L2 정규화 후 KMeans(k). k는 곡 길이로 결정: `k = clip(round(duration / 25), 3, 10)`. 즉 약 25초당 한 클러스터, 최소 3 / 최대 10.
6. **라벨 평활 + 짧은 구간 흡수** — KMeans 라벨에 `median_filter(size=9)`를 한 번 더 걸어 마지막 깜빡임을 제거하고, 라벨이 바뀌는 위치를 경계로 뽑는다. `min_segment_sec(=8.0)` 미만 구간은 다음 우선순위로 인접 구간에 흡수된다:
   1. **같은 라벨인 이웃이 있으면** 그쪽으로 (라벨 일관성 우선)
   2. **한쪽 이웃이 없으면(곡 시작/끝)** 있는 쪽으로
   3. **양쪽 다 다른 라벨이면 더 긴 이웃 쪽으로** (작은 덩어리를 더 큰 덩어리에 흡수)

   "가장 가까운" 데가 아니라 "의미(라벨)가 같은 데" 우선이라는 점이 중요. 한 번 흡수할 때마다 처음부터 다시 스캔(`changed=True` 플래그)해 연쇄 흡수도 깔끔히 처리한다.

#### 주요 파라미터

| 위치 | 이름 | 디폴트 | 의미 |
|---|---|---|---|
| `segment_song` 인자 | `min_segment_sec` | `8.0` | 이보다 짧은 구간은 인접 구간으로 병합. 인트로에서 악기가 하나씩 들어오며 라벨이 깜빡이는 걸 막는다. |
| `chroma_cqt` | `hop_length` | librosa 기본(512) | 시간 해상도. 작을수록 미세하지만 노이즈에 민감. |
| `mfcc` | `n_mfcc` | `13` | 음색 표현의 차원 수. |
| `recurrence_matrix` | `width` | `3` | 같은 위치 근처 비트 끼리는 자기 자신과의 유사도가 너무 높아 의미가 없으니 무시할 인접 폭. |
| `timelag_filter` | `size` | `(1, 7)` | 재발 행렬 대각선 방향 평활 윈도우. 클수록 부드러워지지만 짧은 반복을 놓침. |
| 클러스터 수 식 | `k` | `clip(round(duration/25), 3, 10)` | 곡 25초당 1 클러스터. 30초 미리듣기엔 거의 항상 3, K-pop 한 곡(180~210s)이면 7~8. |
| 라벨 평활 | `median_filter(size=9)` | — | 약 한 마디 분량의 라벨 깜빡임을 흡수. |

#### 실패/주의 모드

- **비트 추출 실패** — 비트가 8개 미만이면 `np.arange(0, n, 8)`로 fallback. 본격적인 의미는 없고 그냥 죽지 않게 하는 안전장치.
- **너무 짧은 곡** — `n < 4`이면 통째로 한 섹션으로 반환.
- **`numpy.linalg.eigh` 실패** — 라플라시안에 `1e-6 * I`를 더해 재시도(degenerate한 affinity일 때).

### 3.3 분위기 경계 — Novelty curve

`mood_novelty_boundaries(mood_frames, window_sec=4.0, min_gap_sec=6.0)` + 그 입력을 만드는 `compute_mood_frames(y, sr, hop=512)`.

#### 알고리즘

1. **프레임 단위 저수준 특징 5종** (`compute_mood_frames`):
   - RMS — 라우드니스 (arousal ↑)
   - Spectral centroid — 밝기 (valence/arousal ↑)
   - Spectral flatness — 텍스처 복잡도/노이즈성 (tension에 기여)
   - Onset strength — 리듬 밀도 (arousal ↑)
   - Tonnetz의 std — 화성 변동성 (tension에 기여)

   여기에 chroma로부터 **로컬 윈도우(약 2초) 안의 major/minor 상관**을 계산해 mode trace를 추가한다. major면 +1, minor면 -1에 가까운 스칼라가 프레임마다 나온다. `_estimate_mode`는 Krumhansl–Schmuckler 키 프로필(`_MAJOR_PROFILE`, `_MINOR_PROFILE`)에 12개 회전 상관 중 최대값을 비교.

2. **z-score 정규화 후 3축으로 합성**:
   ```
   energy     = 0.6 · z(rms) + 0.4 · z(onset)
   brightness = z(centroid)
   valence    = 0.7 · mode_trace + 0.3 · tanh(brightness)
   tension    = 0.5 · z(tonnetz_std) + 0.5 · z(flatness)
   ```
   가중치는 휴리스틱 — MER(music emotion recognition) 논문에서 자주 보고된 valence/arousal 상관자들의 인지된 영향력에 맞춰 임의로 잡았다.

3. **Novelty curve** (`mood_novelty_boundaries`):
   - V = stack(valence, energy, tension)을 시간축으로 좌/우 윈도우로 나눠 각 윈도우의 평균 차이의 L2 norm을 매 프레임 i에서 계산. 윈도우 길이는 `window_sec(=4.0)`초를 프레임 dt로 나눈 만큼.
   - 결과를 0–1로 정규화한 뒤 `scipy.signal.find_peaks(distance=min_gap_sec/dt, prominence=0.2)`로 피크를 뽑고, 그 위치의 시간을 분위기 경계로 반환.

#### 주요 파라미터

| 위치 | 이름 | 디폴트 | 의미 |
|---|---|---|---|
| `compute_mood_frames` | `hop` | `512` | 프레임 hop. 작을수록 시간 해상도 ↑, 노이즈 ↑. |
| 가중치 — energy | `0.6 · rms + 0.4 · onset` | — | 라우드니스가 리듬 밀도보다 약간 우위. |
| 가중치 — valence | `0.7 · mode + 0.3 · tanh(bright)` | — | 키(major/minor)가 밝기보다 우위. |
| 가중치 — tension | `0.5 · tonnetz + 0.5 · flatness` | — | 동등 가중. |
| mode 윈도우 | `int(sr/hop*2)` | 약 2초 | 짧을수록 키 변화에 민감하지만 흔들림이 큼. |
| `mood_novelty_boundaries` | `window_sec` | `4.0` | 좌/우 평균을 잡는 슬라이딩 윈도우 길이. 클수록 큰 분위기 변화만 잡힘. |
|  | `min_gap_sec` | `6.0` | 두 분위기 경계 사이 최소 간격. |
|  | `prominence` | `0.2` | 피크 인정 임계. 작을수록 경계가 많아짐(노이즈 ↑). |

#### 실패/주의 모드

- **`win` 안의 chroma 합이 0에 가까우면** mode estimate는 0으로 폴백 (사실상 무성 구간).
- **곡이 너무 짧으면** (frame 수 < 4) 빈 경계 리스트를 반환.
- **z-score는 곡 단위**라 절대값 비교가 아닌 **곡 내부 상대 비교**로만 의미가 있다 — "이 곡은 valence 0.5"는 그 자체로 의미가 없고, "이 곡 안에서 후렴이 verse보다 valence가 높다" 식으로만 해석.

### 3.4 섹션 요약 (`summarize_sections`) + 라벨링 (`_label_mood`)

구조 경계 안쪽에서 valence/energy/tension의 평균을 내 섹션별 한 줄 요약 dict를 만든다. `_label_mood`가 세 값을 보고 사람이 읽을 수 있는 라벨로 바꾼다:

```
v: bright (>0.2) / dark (<-0.2) / neutral
e: high-energy (>0.3) / calm (<-0.3) / moderate
t: tense (>0.5) / relaxed (<-0.5) / (없음)
→ "high-energy / bright / tense"
```

이 임계값(0.2/0.3/0.5)은 z-score 단위라 곡 내부 상대 비교에 맞춰져 있다.

### 3.5 K-pop 맥락에서의 정성적 트레이드오프

> ⚠️ 이 절은 **체계적인 평가 실험으로 검증한 결과가 아니다.** 곡 구조에 대한 일반적 관찰과 알고리즘 동작에 대한 추론에 기반한 것이며, 본인의 곡 컬렉션으로 직접 확인할 필요가 있다. 이를 위한 절차는 §6에 정리.

**구조 경계가 K-pop에 강한 이유**

- **섹션 사운드 팔레트가 선명하게 갈린다** — 인트로 / 벌스 / 프리코러스 / 코러스 / 댄스 브레이크 / 브릿지가 3분 안에 다 들어가고, 각 섹션에서 악기 편성과 보컬 처리가 통째로 바뀐다. chroma + MFCC가 잡는 패턴이 다른 장르보다 두드러진다.
- **반복 구조가 명확** — A-B-A-B-C-A 형태가 흔해서 라플라시안 분해의 재발 그래프가 잘 동작한다.

**분위기 경계가 자주 false positive를 내는 지점 (추정)**

- 후렴 안의 ad-lib 한 줄, 라스트 코러스의 키 변조 1마디, 보컬 더블링 추가 등 "구조는 그대로인데 4초 평균만 살짝 흔들리는" 변화는 디폴트 `prominence=0.2`로 잘 잡혀버린다.
- 결과적으로 **구조 경계 근처에 + 잡음 몇 개**가 추가된 형태가 되기 쉬움.

**분위기 경계가 정말 유일하게 잡는 지점 (잠재적 가치)**

- **빌드업 → 드롭** 같은 "구조는 같은데 분위기만 변하는 구간". 라플라시안이 같은 클러스터로 묶어버리는 비트들 안에서 valence/energy가 서서히 올라가다 꺾이는 지점.
- 라스트 코러스의 키 업.
- 브릿지 → 라스트 코러스 직전의 호흡(절제된 편곡).

**합의 / 비합의의 해석**

같은 시점에 두 경계가 함께 나오면 → 그 경계는 거의 확실히 의미 있다 (ensemble 효과).
한쪽만 나오면:
- **구조만** → 화성/음색이 바뀌었지만 전체적인 valence/energy/tension은 비슷한 채로 진행 (예: 같은 분위기로 진행되는 verse↔pre-chorus).
- **분위기만** → 화성/음색은 같은 클러스터인데 강도가 변함 (예: 후렴 안의 빌드업 또는 키 업).

### 3.6 분석 결과 내보내기 (`export_result`)

`{stem}_analysis/` 디렉토리에 다음을 떨군다 (Streamlit 탭에선 ZIP으로 묶어 다운로드 제공):

| 파일 | 내용 |
|---|---|
| `{stem}_L.wav` / `{stem}_R.wav` | 원본 L/R 채널 |
| `{stem}_sections.csv` | 섹션 인덱스 / start / end / duration / valence / energy / tension / mood_label |
| `{stem}_pitch.csv` | time / f0_hz / midi / voiced_prob (frame 단위) |
| `{stem}_mood_frames.csv` | time / valence / energy / tension / brightness (frame 단위) |
| `{stem}_boundaries.csv` | 모든 경계 시간 + type (`structural` / `mood` / 둘 다는 `+`로) |
| `{stem}_meta.csv` | duration / sr / tempo_bpm / 피치 요약 |

---

## 4. K-pop Visual Mapping (Visual Mapping 탭)

`process_kpop_visualization(title, artist, audio_path=...)`이 한 곡을 비주얼 컨셉 JSON으로 분류한다. **2단계 분기**가 핵심:

```
Genius에서 곡을 찾으면
  → text 분기: 가사 + Librosa feature를 gemini-2.5-flash로 분류
못 찾으면
  → audio 분기: Librosa feature + 오디오 파일을 gemini-2.5-pro로 분류
```

### 4.1 사용 외부 API

| API | 용도 | 인증 | 모델/엔드포인트 |
|---|---|---|---|
| **iTunes Search API** | 곡 메타+30초 미리듣기 URL | 불필요 (rate limit만 있음) | `https://itunes.apple.com/search` |
| **Genius API** (lyricsgenius) | 가사 + 곡 메타 검색 | `GENIUS_ACCESS_TOKEN` | `Genius.search_song()` |
| **Google Gemini** (`google-genai`) | 텍스트 또는 오디오 멀티모달 분류 | `GEMINI_API_KEY` | text=`gemini-2.5-flash`, audio=`gemini-2.5-pro` |
| **yt-dlp** | (현재 입력 폼에선 비활성) YouTube 풀 트랙 | 불필요 | `bestaudio[ext=m4a]/bestaudio` |

iTunes/YouTube 음원 확보 함수는 살아있지만, **Streamlit 입력 폼은 사용자 업로드 파일만 받으므로 `audio_path`를 직접 넘겨 그 두 fallback을 우회**한다. 노트북에서 `acquire_audio()`를 직접 호출하면 iTunes → YouTube 순으로 받아온다.

### 4.2 캐시 (`kpop_visual_db.json`)

곡 단위로 분류 결과를 영구 저장. 키는 `f"{title.lower().strip()}__{artist.lower().strip()}"`. `process_kpop_visualization`은 시작할 때 캐시를 보고 히트면 즉시 반환, force_refresh=True면 무시. Streamlit "캐시 DB" 탭에서 캐시된 항목을 다시 열람 가능.

엔트리 구조:

```jsonc
{
  "metadata": {
    "title": "Supernova",
    "artist": "aespa",
    "branch": "known",            // "known" = Genius hit, "new" = Gemini audio
    "genius_url": "...",          // known 분기에만
    "audio_source": "aespa_Supernova.m4a"
  },
  "live_features": {              // Librosa 정량 feature (§4.3)
    "duration": 30.0, "tempo": 117.45,
    "energy": 1.0, "valence": 0.34, "danceability": 0.76,
    "key": 4, "harmonic_ratio": 0.67, "dynamics_range": 4.09
  },
  "concept_features": {           // Gemini가 생성한 비주얼 컨셉 (§4.6)
    "sonic_texture": "Synth-pop",
    "narrative_archetype": "Cyberpunk",
    "visual_symbols": ["Supernova", "Tick-tick bomb", "Portal"],
    "color_palette": {"main": "#8A2BE2", "sub": ["#FFD700", "#000033"]},
    "reasoning_brief": "..."
  }
}
```

### 4.3 정량 feature 추출 (`extract_audio_features`)

Librosa로 8개 단순 통계값을 뽑는다 — 이건 §3의 frame 단위 분석과는 별개이고, **Gemini에 prompt context로 던져주는 요약값**이다.

| 키 | 계산 |
|---|---|
| `duration` | `librosa.get_duration` |
| `tempo` | `librosa.beat.beat_track`의 BPM |
| `energy` | `min(1, mean(RMS) * 8)` — 0–1로 클립 |
| `valence` | `clip((mean(centroid) - 1000) / 4000, 0, 1)` — Spotify 식 단순 매핑 |
| `danceability` | `clip(1 - mean(zcr) * 3, 0, 1)` — zero-crossing rate 기반 |
| `key` | `chroma_stft.mean(axis=1).argmax()` — 0=C, 1=C#, … 11=B |
| `harmonic_ratio` | `mean(|harmonic|) / (mean(|harmonic|) + mean(|percussive|))` from HPSS |
| `dynamics_range` | `onset_env`의 95th – 5th percentile |

이 값들은 §3의 z-score 기반 분석과 의미가 다르다 — 여기선 절대값 단순 매핑이라 곡 간 비교가 가능하다는 점이 장점.

### 4.4 Genius 분기 (`process_known_song`)

1. `genius.search_song(title, artist)` — `lyricsgenius`가 Genius API에 검색 요청
2. 검색 hit이면 가사를 가져와 첫 1500자만 잘라서 narrative text 구성:
   ```
   Title: ...
   Artist: ...
   Album: ...
   Lyrics excerpt:
   ...
   ```
3. `gemini_classify_text(features, narrative_text)`으로 분류

### 4.5 Gemini Audio 분기 (`process_new_song`)

Genius miss(신곡 / 인디 / 영어권 외 잘 안 잡힘) 시 `gemini-2.5-pro`에 오디오 파일을 직접 첨부:

```python
contents=[
    types.Part.from_bytes(data=audio_bytes, mime_type=...),
    f"Song: {title} - {artist}\nLibrosa features: {...}\n"
    "가사/메타가 없는 신곡이다. 첨부된 오디오를 직접 들어 분류하라."
]
```

mime은 `_audio_mime()`이 확장자로 결정 (m4a/mp4 → audio/mp4, wav → audio/wav 등).

### 4.6 분류 표준 (SYSTEM_PROMPT)

분류는 두 분기 모두 같은 SYSTEM_PROMPT를 사용한다. 일관된 K-pop 비주얼 어휘를 강제하려는 것:

**Sonic Texture (8개 중 1개 선택)**
> Synth-pop · Brass-Heavy · Glitch · Orchestral · Acoustic · Rock-chic · Hip-hop Beat · Dreamy Pad

**Narrative Archetype (7개 중 1개 선택)**
> High-teen · Cyberpunk · Ethereal · Narcissistic · Mala-taste (Spicy) · Retro-nostalgia · Gothic-horror

**Visual Symbol** — 자유 텍스트 3개 (동물/사물/자연물 등)
**Color Palette** — 메인 hex 1개 + 서브 hex 2개. valence/energy를 명도/채도에 반영하라고 명시적으로 지시.
**Reasoning brief** — 한글 2-3문장.

응답은 `response_mime_type="application/json"` + `temperature=0.3`로 강제 JSON. `_parse_json_response`가 첫 번째 `{...}` 블록을 정규식으로 잘라 `json.loads`. 마크다운 코드 펜스가 섞여도 통과.

---

## 5. 시각화

### 5.1 WaveSurfer.js 인터랙티브 플레이어

`_wavesurfer_player()`가 `streamlit.components.v1.html`로 HTML을 임베드. WaveSurfer.js v7.8.6을 unpkg CDN에서 로드 + Regions 플러그인 + Timeline 플러그인. 오디오는 base64 data URL로 inline.

기능:
- 파형 위에 섹션 색칠 (구조 경계 기반, tab20 팔레트)
- 빨간 마커 region = 구조 경계, 파란 마커 = 분위기 경계
- 파형/region/하단 섹션 바 어디든 클릭하면 해당 시점으로 점프
- 재생 커서가 자동으로 따라감
- 재생/정지, 처음으로 버튼, `1:23 / 3:45` 시간 표시

### 5.2 Plotly 3행 차트

`_plotly_analysis()` — 정적이지만 인터랙티브:
- **Row 1**: 파형 (4000pt로 다운샘플) + 섹션 음영 + 라벨 annotation
- **Row 2**: 피치 contour (log Hz)
- **Row 3**: Mood novelty curve (fill area)
- 모든 row에 구조/분위기 경계 vline (토글에 따라 표시 여부 결정)
- 줌·팬·hover 모두 가능 (`hovermode="x unified"`)

### 5.3 matplotlib 백업

`_plot_analysis()` — Streamlit expander 안에 접어둠. PNG로 export하기 좋은 정적 그림이 필요할 때.

---

## 6. 본인이 직접 해볼 수 있는 것 — 실험 가이드

여기까지가 코드와 알고리즘 사실이고, 이 절은 본인이 본인 곡 컬렉션으로 검증할 수 있는 실험들이다. 평가용 ground truth가 없는 상태라 정량 평가는 어렵고, 정성적 비교가 주가 된다.

### 6.1 파라미터 sweep — 어디까지 줄이고 늘려도 결과가 안정적인가

**구조 경계 쪽**

- `min_segment_sec`을 4 / 8 / 12 / 16으로 흔들며 boundary 개수의 변화를 보기. K-pop 평균 섹션이 12~20초임을 감안하면 8보다 큰 값이 적절할 수도 있음.
- 클러스터 수 공식 `duration/25`의 25를 15 / 25 / 35로 흔들기. K-pop 한 곡에 7~8개 클러스터가 적당해 보이지만 본인 곡에선 다를 수 있음.
- `recurrence_matrix(width=...)`를 1 / 3 / 5로. width가 클수록 곡 안 멀리 떨어진 비슷한 부분만 연결됨.

**분위기 경계 쪽**

- `prominence`를 0.1 / 0.2 / 0.35 / 0.5로. **이게 분위기 경계의 가장 큰 변수.** 0.5쯤이면 정말 큰 변화만 남고, 0.1이면 후렴 안의 미세 변화도 다 잡힘.
- `window_sec`을 2 / 4 / 8로. 윈도우가 크면 점진적 빌드업도 한 피크로 합쳐짐.
- `min_gap_sec`을 4 / 6 / 10으로. 분위기 경계가 너무 촘촘하게 붙어 나올 때 상한선 역할.

**가중치 흔들기** (코드 직접 수정 필요)

- `valence = 0.7 · mode + 0.3 · tanh(bright)`에서 mode 가중치를 0.4–0.9 범위로. K-pop은 major/minor 외에 모달 변환이 흔해서 mode 가중치가 클수록 잡음이 늘 가능성.
- `energy`의 RMS:onset 비율을 0.5:0.5 ~ 0.8:0.2로. 빌드업 같은 "라우드니스 변화 없이 리듬만 풍성해지는 구간"이 잡히는지 확인.

### 6.2 합의/비합의 통계 — 두 경계가 얼마나 같은 시점을 가리키는가

같은 곡에 대해 구조 경계와 분위기 경계의 시간 리스트를 뽑은 뒤, 각 분위기 경계에 대해 가장 가까운 구조 경계까지의 시간 차를 계산하면 분포가 나온다.

- 차이가 1~2초 안에 몰려있다 → **두 검출의 정보 중복이 크다.** 분위기 경계만 따로 살릴 가치가 거의 없음.
- 차이가 10초 이상에 꼬리가 길다 → **분위기가 정말 유일하게 잡는 지점이 있다.** 그게 무엇인지 직접 들어보고 가치를 판단.

곡 5~10개로 이걸 돌려보면 본인 컬렉션 기준의 답이 나옴.

### 6.3 Demucs 프리셋 평가

Demucs 4-stem은 모델 자체를 바꾸지 않는 한 결과가 같지만, 프리셋 합성 단계에서 다음을 시도해볼 만:

- **현재 "멜로디"의 정의**: `other`만 = 기타/피아노/신스. `bass + other`도 한 옵션. K-pop에서 베이스가 멜로디 역할도 하는 곡(특히 R&B 영향권)이라면 후자가 더 자연스러울 수 있음. 곡에 따라 프리셋을 바꿔가며 들어보고 어느 정의가 본인 작업에 더 유용한지 결정.
- **Demucs 모델 변경**: htdemucs 외에 `htdemucs_ft`(fine-tuned, 약간 더 깨끗함, 4배 느림), `mdx_extra_q`(가벼움)도 있음. `load_demucs_model("htdemucs_ft")`로 한 줄만 바꿔 비교.

### 6.4 Visual Mapping 평가 — 같은 곡 여러 번 분류

`temperature=0.3`이라 결정적은 아니다. 같은 곡에 `force_refresh=True`로 5회 호출해서 sonic_texture / narrative_archetype 분포를 보면 모델의 **확신도**가 드러난다.

- 5번 다 같은 라벨 → 분류가 명확
- 2~3개 라벨로 갈림 → 곡 자체가 경계에 있음 (예: Synth-pop ↔ Hip-hop Beat 모호)
- 매번 다른 라벨 → SYSTEM_PROMPT의 기준이 그 곡 유형에는 잘 안 맞음 → 분류 표준 확장 필요

분류 표준 자체(8 sonic textures, 7 archetypes)는 K-pop 비주얼 어휘에 맞춘 임의 선택이라, 본인 작업이 다른 어휘를 쓴다면 `SONIC_TEXTURES` / `NARRATIVE_ARCHETYPES`를 직접 수정.

### 6.5 ground truth가 있다면 — 본격 평가

직접 라벨링한 boundary 시간 리스트가 있다면 MIREX의 표준 지표 두 개가 적용 가능:

- **Hit rate (F-measure with tolerance)** — ground truth 경계와 ±0.5s(또는 ±3s) 안에 검출된 경계가 들어오면 hit으로 보고 P/R/F 계산. mir_eval 라이브러리가 한 줄로 해준다 (`mir_eval.segment.detection`).
- **Boundary deviation** — 매칭된 hit pair의 시간 차 분포(median, 90th percentile).

K-pop 곡 10개 정도면 의미 있는 비교가 되고, 100개쯤 라벨링하면 서로 다른 파라미터 셋의 우열을 통계적으로 결론낼 수 있다.

### 6.6 시각적 디버깅 워크플로

1. Streamlit 앱에서 곡을 분석하고
2. WaveSurfer 플레이어 + Plotly novelty 차트를 띄운 채
3. 토글을 "둘 다"로 두고 각 경계가 어디 떨어지는지 들으며 확인
4. 이상한 경계가 보이면 그 부근의 mood frame 값(valence/energy/tension의 raw trace)을 확인 — 무엇이 그 novelty 피크를 만들었는지 역추적

이 사이클을 5~10곡 돌리면 디폴트 파라미터가 본인 컬렉션에 맞는지 감이 잡힌다.

---

## 7. 환경 설정

### 7.1 의존성

`requirements.txt`:

| 카테고리 | 패키지 |
|---|---|
| UI | `streamlit` |
| 오디오 I/O | `numpy`, `soundfile`, `librosa` |
| 신호 처리 / ML | `scipy`, `scikit-learn` |
| 시각화 | `matplotlib`, `plotly` |
| 트랙 분리 | `torch`, `demucs` |
| Visual Mapping | `python-dotenv`, `requests`, `lyricsgenius`, `google-genai`, `yt-dlp` |

ffmpeg는 시스템에 별도 설치 필요 (mp3/m4a 디코딩용 — `brew install ffmpeg`).

### 7.2 환경 변수 (`.env`)

```
GENIUS_ACCESS_TOKEN=...   # https://genius.com/api-clients
GEMINI_API_KEY=...        # https://aistudio.google.com/apikey
```

`.env.example`을 복사해 채우면 된다. Visual Mapping을 안 쓸 거면 둘 다 비어 있어도 음악 분석 / 음원 분리 탭은 정상 동작.

### 7.3 실행

```bash
streamlit run streamlit_app.py
```

---

## 8. 참고 문헌

- McFee, B., & Ellis, D. P. W. (2014). *Analyzing song structure with spectral clustering.* ISMIR.
- Krumhansl, C. L. (1990). *Cognitive foundations of musical pitch.* (mode estimation의 토대 — `_MAJOR_PROFILE` / `_MINOR_PROFILE` 값.)
- Défossez, A. (2021). *Hybrid Spectrogram and Waveform Source Separation.* (Demucs v3/v4의 base 논문.)
- mir_eval — `mir_eval.segment.detection` (boundary F-measure 표준 구현).
