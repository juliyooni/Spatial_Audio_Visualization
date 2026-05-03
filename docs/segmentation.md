# 음악 구조/분위기 경계 감지 — 기술 문서

`music_analysis/pipeline.py`는 한 곡에서 두 종류의 시간축 경계를 따로 뽑는다.

| 경계 종류 | 보는 것 | 답하는 질문 | 주된 신호 |
|---|---|---|---|
| **구조 경계** | 화성·음색의 반복 패턴 | "벌스/코러스가 어디서 반복되는가?" | chroma + MFCC |
| **분위기 경계** | valence/energy/tension의 시간적 변화 | "분위기가 언제 바뀌는가?" | RMS, spectral centroid, mode, tonnetz |

둘은 합쳐지지 않고 나란히 반환된다 — `result["segmentation"]["boundary_times"]`와 `result["mood"]["mood_boundaries"]["times"]`. Streamlit 앱에서는 토글로 둘 중 하나만 보거나 둘 다 겹쳐 보는 식으로 비교할 수 있다.

## 1. 구조 경계 — 라플라시안 스펙트럴 분해

`segment_song(y, sr, min_segment_sec=8.0)` — McFee & Ellis (2014).

### 알고리즘

1. **비트 동기 특징** — `librosa.beat.beat_track`으로 비트를 잡고, chroma(화성)와 MFCC(음색) 13차를 비트 단위로 묶는다(`librosa.util.sync`). 이후 모든 거리 계산이 프레임이 아니라 비트 기준이라 텀포에 의존하지 않는다.
2. **재발 그래프(R)** — chroma 비트 시퀀스에서 `librosa.segment.recurrence_matrix(width=3, mode="affinity")`로 만든 affinity 행렬. 이게 "곡 멀리 떨어진 두 비트가 화성적으로 얼마나 비슷한가"를 잡는다. 라벨 깜빡임을 줄이려고 `timelag_filter(median_filter, size=(1,7))`로 대각선 방향 평활.
3. **국소 유사도(R_path)** — 인접 비트 간 MFCC L2 거리에 `exp(-d/σ)`를 씌워 인접한 두 비트가 음색적으로 얼마나 이어지는지 표현. 곡의 흐름을 끊지 않게 하는 역할.
4. **그래프 결합** — McFee의 balancing μ로 `A = μ·R + (1-μ)·R_path`. 두 그래프의 degree 벡터로 계산해 둘의 영향력을 자동 균형.
5. **정규화 라플라시안 → 고유분해 → KMeans** — `csgraph_laplacian(A, normed=True)`의 첫 k개 고유벡터를 행 단위 L2 정규화 후 KMeans(k). k는 곡 길이로 결정: `k = clip(round(duration / 25), 3, 10)`. 즉 약 25초당 한 클러스터, 최소 3 / 최대 10.
6. **라벨 평활 + 짧은 구간 흡수** — KMeans 라벨에 `median_filter(size=9)`를 한 번 더 걸어 마지막 깜빡임을 제거하고, 라벨이 바뀌는 위치를 경계로 뽑는다. `min_segment_sec(=8.0)` 미만 구간은 인접 구간 중 같은 라벨 쪽 → 짧은 쪽 → 긴 쪽 순으로 흡수한다.

### 주요 파라미터

| 위치 | 이름 | 디폴트 | 의미 |
|---|---|---|---|
| `segment_song` 인자 | `min_segment_sec` | `8.0` | 이보다 짧은 구간은 인접 구간으로 병합. 인트로에서 악기가 하나씩 들어오며 라벨이 깜빡이는 걸 막는다. |
| `chroma_cqt` | `hop_length` | librosa 기본(512) | 시간 해상도. 작을수록 미세하지만 노이즈에 민감. |
| `mfcc` | `n_mfcc` | `13` | 음색 표현의 차원 수. |
| `recurrence_matrix` | `width` | `3` | 같은 위치 근처 비트 끼리는 자기 자신과의 유사도가 너무 높아 의미가 없으니 무시할 인접 폭. |
| `timelag_filter` | `size` | `(1, 7)` | 재발 행렬 대각선 방향 평활 윈도우. 클수록 부드러워지지만 짧은 반복을 놓침. |
| 클러스터 수 식 | `k` | `clip(round(duration/25), 3, 10)` | 곡 25초당 1 클러스터. 30초 미리듣기엔 거의 항상 3, K-pop 한 곡(180~210s)이면 7~8. |
| 라벨 평활 | `median_filter(size=9)` | — | 약 한 마디 분량의 라벨 깜빡임을 흡수. |

### 실패/주의 모드

- **비트 추출 실패** — 비트가 8개 미만이면 `np.arange(0, n, 8)`로 fallback. 본격적인 의미는 없고 그냥 죽지 않게 하는 안전장치.
- **너무 짧은 곡** — `n < 4`이면 통째로 한 섹션으로 반환.
- **`numpy.linalg.eigh` 실패** — 라플라시안에 `1e-6 * I`를 더해 재시도(degenerate한 affinity일 때).

---

## 2. 분위기 경계 — Novelty curve

`mood_novelty_boundaries(mood_frames, window_sec=4.0, min_gap_sec=6.0)` + 그 입력을 만드는 `compute_mood_frames(y, sr, hop=512)`.

### 알고리즘

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

### 주요 파라미터

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

### 실패/주의 모드

- **`win` 안의 chroma 합이 0에 가까우면** mode estimate는 0으로 폴백 (사실상 무성 구간).
- **곡이 너무 짧으면** (frame 수 < 4) 빈 경계 리스트를 반환.
- **z-score는 곡 단위**라 절대값 비교가 아닌 **곡 내부 상대 비교**로만 의미가 있다 — "이 곡은 valence 0.5"는 그 자체로 의미가 없고, "이 곡 안에서 후렴이 verse보다 valence가 높다" 식으로만 해석.

---

## 3. K-pop 맥락에서의 정성적 트레이드오프

> ⚠️ 이 절은 **체계적인 평가 실험으로 검증한 결과가 아니다.** 곡 구조에 대한 일반적 관찰과 알고리즘 동작에 대한 추론에 기반한 것이며, 본인의 곡 컬렉션으로 직접 확인할 필요가 있다. 이를 위한 절차는 §4에 정리.

### 구조 경계가 K-pop에 강한 이유

- **섹션 사운드 팔레트가 선명하게 갈린다** — 인트로 / 벌스 / 프리코러스 / 코러스 / 댄스 브레이크 / 브릿지가 3분 안에 다 들어가고, 각 섹션에서 악기 편성과 보컬 처리가 통째로 바뀐다. chroma + MFCC가 잡는 패턴이 다른 장르보다 두드러진다.
- **반복 구조가 명확** — A-B-A-B-C-A 형태가 흔해서 라플라시안 분해의 재발 그래프가 잘 동작한다.

### 분위기 경계가 자주 false positive를 내는 지점 (추정)

- 후렴 안의 ad-lib 한 줄, 라스트 코러스의 키 변조 1마디, 보컬 더블링 추가 등 "구조는 그대로인데 4초 평균만 살짝 흔들리는" 변화는 디폴트 `prominence=0.2`로 잘 잡혀버린다.
- 결과적으로 **구조 경계 근처에 + 잡음 몇 개**가 추가된 형태가 되기 쉬움.

### 분위기 경계가 정말 유일하게 잡는 지점 (잠재적 가치)

- **빌드업 → 드롭** 같은 "구조는 같은데 분위기만 변하는 구간". 라플라시안이 같은 클러스터로 묶어버리는 비트들 안에서 valence/energy가 서서히 올라가다 꺾이는 지점.
- 라스트 코러스의 키 업.
- 브릿지 → 라스트 코러스 직전의 호흡(절제된 편곡).

이런 지점이 본인 곡에서 의미가 있다면 분위기 경계도 살릴 가치가 있다. 그렇지 않다면 구조 경계 단일로 충분.

### 합의 / 비합의의 해석

같은 시점에 두 경계가 함께 나오면 → 그 경계는 거의 확실히 의미 있다 (ensemble 효과).
한쪽만 나오면:
- **구조만** → 화성/음색이 바뀌었지만 전체적인 valence/energy/tension은 비슷한 채로 진행 (예: 같은 분위기로 진행되는 verse↔pre-chorus).
- **분위기만** → 화성/음색은 같은 클러스터인데 강도가 변함 (예: 후렴 안의 빌드업 또는 키 업).

---

## 4. 본인이 직접 해볼 수 있는 것 — 실험 가이드

여기까지가 코드와 알고리즘 사실이고, 이 절은 본인이 본인 곡 컬렉션으로 검증할 수 있는 실험들이다. 평가용 ground truth가 없는 상태라 정량 평가는 어렵고, 정성적 비교가 주가 된다.

### 4.1 파라미터 sweep — 어디까지 줄이고 늘려도 결과가 안정적인가

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

### 4.2 합의/비합의 통계 — 두 경계가 얼마나 같은 시점을 가리키는가

같은 곡에 대해 구조 경계와 분위기 경계의 시간 리스트를 뽑은 뒤, 각 분위기 경계에 대해 가장 가까운 구조 경계까지의 시간 차를 계산하면 분포가 나온다.

- 차이가 1~2초 안에 몰려있다 → **두 검출의 정보 중복이 크다.** 분위기 경계만 따로 살릴 가치가 거의 없음.
- 차이가 10초 이상에 꼬리가 길다 → **분위기가 정말 유일하게 잡는 지점이 있다.** 그게 무엇인지 직접 들어보고 가치를 판단.

곡 5~10개로 이걸 돌려보면 본인 컬렉션 기준의 답이 나옴.

### 4.3 ground truth가 있다면 — 본격 평가

직접 라벨링한 boundary 시간 리스트가 있다면 MIREX의 표준 지표 두 개가 적용 가능:

- **Hit rate (F-measure with tolerance)** — ground truth 경계와 ±0.5s(또는 ±3s) 안에 검출된 경계가 들어오면 hit으로 보고 P/R/F 계산. mir_eval 라이브러리가 한 줄로 해준다 (`mir_eval.segment.detection`).
- **Boundary deviation** — 매칭된 hit pair의 시간 차 분포(median, 90th percentile).

K-pop 곡 10개 정도면 의미 있는 비교가 되고, 100개쯤 라벨링하면 서로 다른 파라미터 셋의 우열을 통계적으로 결론낼 수 있다.

### 4.4 시각적 디버깅 워크플로

1. Streamlit 앱에서 곡을 분석하고
2. WaveSurfer 플레이어 + Plotly novelty 차트를 띄운 채
3. 토글을 "둘 다"로 두고 각 경계가 어디 떨어지는지 들으며 확인
4. 이상한 경계가 보이면 그 부근의 mood frame 값(valence/energy/tension의 raw trace)을 확인 — 무엇이 그 novelty 피크를 만들었는지 역추적

이 사이클을 5~10곡 돌리면 디폴트 파라미터가 본인 컬렉션에 맞는지 감이 잡힌다.

---

## 5. 참고 문헌

- McFee, B., & Ellis, D. P. W. (2014). *Analyzing song structure with spectral clustering.* ISMIR.
- Krumhansl, C. L. (1990). *Cognitive foundations of musical pitch.* (mode estimation의 토대 — `_MAJOR_PROFILE` / `_MINOR_PROFILE` 값.)
- mir_eval — `mir_eval.segment.detection` (boundary F-measure 표준 구현).
