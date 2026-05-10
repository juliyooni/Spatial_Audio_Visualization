# Spatial Audio Visualization — K-pop 통합 분석 앱

K-pop 한 곡을 업로드하면 **음악 분석 / 음원 분리 / 비주얼 매핑**을 한 번에 돌리고 결과를 탭으로 비교해볼 수 있는 Streamlit 앱.

상세한 알고리즘 설명은 [docs/pipeline.md](docs/pipeline.md) 참고.

---

## 한눈에

```
[ 입력 폼 ]
음원 파일 + 제목 + 아티스트 + 실행할 파이프라인 체크박스
        │
        ▼
선택된 파이프라인을 순차 실행
        │
        ├── 🎼 음악 분석   — pitch / 구조 경계 / 분위기 / 통합 섹셔닝
        ├── 🎧 음원 분리   — 채널(L/R/Mid/Side) + Demucs 4-stem
        └── 🌌 Visual Mapping — Librosa + Genius + Gemini → 비주얼 컨셉 JSON
        │
        ▼
탭 4개에서 결과 확인 (음악 분석 · 음원 분리 · Visual Mapping · 캐시 DB)
```

같은 업로드 한 번으로 셋 다 일관된 결과를 얻는 게 핵심.

---

## 디렉토리 구조

```
Spatial_Audio_Visualization/
│
├── streamlit_app.py              # 앱 본체 (입력 폼 + 결과 탭들)
├── requirements.txt              # Python 의존성
├── packages.txt                  # Streamlit Cloud용 apt 패키지 (ffmpeg, libsndfile1)
│
├── music_analysis/
│   ├── pipeline.py               # 피치 / 구조 / 분위기 / 통합 섹셔닝
│   ├── visual_mapping.py         # iTunes/YouTube 음원 + Genius + Gemini
│   ├── kpop_visual_db.json       # Visual Mapping 결과 캐시 (DB)
│   └── *.ipynb                   # 알고리즘 프로토타입 노트북
│
├── docs/
│   └── pipeline.md               # 파이프라인 기술 문서
│
└── samples/                      # 샘플 오디오 (gitignored, .gitkeep만 추적)
```

---

## 주요 기능

### 🎼 음악 분석 (`music_analysis/pipeline.py`)
- **피치**: pYIN으로 f0 / MIDI / voiced 추정
- **구조 경계**: chroma + MFCC 기반 라플라시안 세그멘테이션 (McFee & Ellis 2014). 비트 정렬.
- **분위기**: 프레임마다 valence / energy / tension 산출 → novelty peak로 분위기 전환부 검출
- **통합 섹셔닝**: 두 종류 경계를 ±3초 내에서 묶어 하나의 섹션 트랙으로 합성. 도입부 0~3초는 경계 제외.

### 🎧 음원 분리
- **채널 분리**: L / R / Mid / Side (즉시)
- **트랙 분리**: Demucs `htdemucs` 4-stem(보컬/드럼/베이스/기타) — Apple Silicon에선 MPS, 그 외 CPU
- 결과 탭에서 **프리셋 토글**(보컬+멜로디, 비트만, 반주만 등) 가능

### 🌌 Visual Mapping
가사 유무에 따라 두 분기로 라우팅:
- **known 분기** — Genius 가사 있음 → `gemini-2.5-flash`에 가사+Librosa 피처 전달
- **new 분기** — 가사 없는 신곡 → `gemini-2.5-pro`에 오디오 파일 통째로 첨부

출력 JSON: `sonic_texture / narrative_archetype / visual_symbols / color_palette / reasoning_brief` — 모두 사전 정의된 분류 라벨로 일관성 유지.

---

## 실행

### 사전 요구사항
- Python 3.11+
- ffmpeg (`brew install ffmpeg`)
- API 키 두 개: [Genius](https://genius.com/api-clients), [Gemini](https://aistudio.google.com/app/apikey)

### 로컬 개발

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# .env에 키 작성 (gitignored)
cat > .env <<EOF
GENIUS_ACCESS_TOKEN=...
GEMINI_API_KEY=...
EOF

streamlit run streamlit_app.py
```

### Streamlit Community Cloud 배포

`.streamlit/secrets.toml`(gitignored) 또는 앱 Settings → Secrets에 다음을 등록:

```toml
GENIUS_ACCESS_TOKEN = "..."
GEMINI_API_KEY = "..."
```

`load_env`가 `st.secrets → os.environ → .env` 순으로 폴백하므로 둘 다 자동으로 인식.

---

## 데이터 캐시

- **Visual Mapping 결과**: [music_analysis/kpop_visual_db.json](music_analysis/kpop_visual_db.json)에 `(title, artist)` 정규화 키로 저장. 같은 곡 재요청 시 Gemini 호출 안 함.
- **음원**: iTunes 미리듣기(30초)를 1순위, 실패 시 사용자가 준 YouTube URL을 yt-dlp로 받아옴.

---

## 내보내기 (CSV/WAV ZIP)

음악 분석 탭 하단의 **ZIP 생성** 버튼:
- `*_L.wav` / `*_R.wav` — 채널 분리 음원
- `*_sections.csv` — **통합 섹션** (있을 때) 또는 mood 섹션. valence/energy/tension/mood_label/boundary_sources 포함
- `*_pitch.csv` — 시간별 f0/midi/voiced
- `*_mood_frames.csv` — 시간별 valence/energy/tension/brightness
- `*_boundaries.csv` — 구조/분위기 경계 raw 시각

---

## 라이센스 / 출처

- 구조 세그멘테이션: McFee, B., & Ellis, D. P. W. (2014). *Analyzing song structure with spectral clustering.* ISMIR.
- Demucs: Défossez, A. (2021). *Hybrid Spectrogram and Waveform Source Separation.*
