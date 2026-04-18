# 7.1.4 Spatial Audio Visualization System

7.1.4 채널 공간 음향 데이터를 활용한 **StyleGAN3 기반 영상 생성 시스템**.
사용자가 레퍼런스 이미지와 7.1.4 오디오를 업로드하면, 12개 채널의 공간 좌표를 설정하고, 오디오 에너지에 반응하는 영상을 생성합니다.

---

## 시스템 구조

```
Spatial_Audio_Visualization/
│
├── src/                          # 프론트엔드 (React + Vite)
│   ├── main.jsx                  # 엔트리포인트
│   ├── index.css                 # Tailwind CSS
│   ├── App.jsx                   # 워크플로우 상태 머신 (Upload→Setup→Bake→Play)
│   └── components/
│       ├── UploadStep.jsx        # 이미지/오디오 파일 업로드 UI
│       ├── SetupStep.jsx         # 5스크린 프리뷰 + 12채널 마커 드래그
│       ├── ScreenLayout.jsx      # L2/L1/CENTER/R1/R2 스크린 배치
│       ├── MarkerOverlay.jsx     # 12개 드래그 가능 마커 (Framer Motion)
│       ├── BakeStep.jsx          # Bake 진행률 UI (시뮬레이션 / SSE 모드)
│       └── PlayStep.jsx          # 결과 영상 재생 + 12채널 실시간 레벨 미터
│
├── backend/                      # 백엔드 (FastAPI + PyTorch)
│   ├── main.py                   # FastAPI 앱 + CORS + MPS 메모리 설정
│   ├── config.py                 # 디바이스/경로/오디오/비디오 설정값
│   ├── requirements.txt          # Python 의존성
│   ├── run.sh                    # 서버 시작 스크립트
│   ├── api/
│   │   └── routes.py             # REST API 엔드포인트 정의
│   ├── core/
│   │   ├── audio_analyzer.py     # 12채널 오디오 → RMS/Onset 분석 (Librosa)
│   │   ├── spatial_modulator.py  # 좌표+RMS → Gaussian Energy Map → Warp Field
│   │   ├── stylegan3_engine.py   # StyleGAN3 모델 로드 + 텍스처 생성
│   │   └── video_baker.py        # 5스크린 합성 + 이펙트 + FFmpeg 인코딩
│   ├── models/                   # StyleGAN3 .pkl 모델 (자동 다운로드)
│   ├── uploads/                  # 업로드된 파일 임시 저장
│   └── outputs/                  # 생성된 영상 파일
│
├── index.html                    # Vite HTML
├── vite.config.js                # Vite + Tailwind 설정
├── package.json                  # Node.js 의존성
└── .gitignore
```

**총 코드**: ~1,880줄 (프론트 830줄 + 백엔드 1,050줄)

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| 프론트엔드 | React 19, Vite 8, Tailwind CSS 4, Framer Motion |
| 백엔드 | Python 3.11, FastAPI, Uvicorn |
| AI/ML | PyTorch (MPS), NVIDIA StyleGAN3 (`stylegan3-t-metfaces-1024x1024.pkl`) |
| 오디오 분석 | Librosa, SoundFile |
| 영상 인코딩 | FFmpeg (H.264/AAC) |
| 통신 | REST API + SSE (Server-Sent Events) |

---

## 워크플로우 (4단계)

### Step 1: Upload
- 사용자가 **레퍼런스 이미지**(PNG/JPG)와 **7.1.4 오디오 파일**(WAV/FLAC/MP3)을 업로드
- 파일 객체(File)와 blob URL을 모두 보존하여 백엔드 전송 및 프리뷰에 사용

### Step 2: Setup
- 레퍼런스 이미지를 **5개 스크린**으로 배치하여 프리뷰:
  - `L2`, `L1`: 원본 이미지 좌측 50% 영역 크롭
  - `CENTER`: 원본 전체 이미지 (16:9 대형)
  - `R1`, `R2`: 원본 이미지 우측 50% 영역 크롭
- **12개 채널 마커**(L, R, C, LFE, Ls, Rs, Lrs, Rrs, Ltf, Rtf, Ltr, Rtr)를 드래그하여 시각적 발원지 설정
- 각 마커의 좌표는 **0.0~1.0 정규화**되어 저장
- **Flat/7.1.4 모드 토글**: Flat 모드 시 모든 마커가 중앙으로 수렴 (공간감 제거 시각화)

### Step 3: Bake
- 프론트에서 `POST /api/bake`로 이미지 + 오디오 + 채널 좌표 JSON 전송
- 백엔드에서 SSE 스트림으로 진행률 실시간 전송
- 파이프라인:
  1. StyleGAN3 모델 로드 (CPU, 자동 다운로드)
  2. Seed → W 벡터 계산 (CPU, instant)
  3. 12채널 오디오 RMS/Onset 분석 (Librosa)
  4. W-space 시퀀스 생성 (누적 드리프트 + 에너지 점프)
  5. 프레임별 생성:
     - 5스크린 레이아웃 합성 (1920x1080)
     - 채널별 방사형 왜곡 (radial warp)
     - 채널별 색상 glow 이펙트
     - **StyleGAN3 텍스처 블렌딩** (채널 색상 틴트 + 에너지 비례 강도)
     - 채널 마커 + 에너지 링 오버레이
  6. FFmpeg H.264 인코딩 (오디오 스테레오 다운믹스 포함)

### Step 4: Play
- 생성된 MP4 영상 재생 (`GET /api/video/{name}`)
- **Web Audio API** AnalyserNode로 12채널 실시간 레벨 미터 표시
- 채널 좌표 JSON 출력 및 클립보드 복사 기능

---

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/health` | 서버 상태 + GPU 백엔드 확인 |
| `POST` | `/api/analyze` | 12채널 오디오 → 채널별 RMS/Onset JSON |
| `POST` | `/api/bake` | 영상 생성 (SSE 스트림으로 진행률 전송) |
| `GET` | `/api/video/{filename}` | 생성된 영상 파일 서빙 |

### `/api/bake` 요청 형식

```
Content-Type: multipart/form-data

Fields:
  image: File (레퍼런스 이미지)
  audio: File (7.1.4 채널 오디오)
  channel_positions: JSON string
    예: {"L": {"x": 0.25, "y": 0.45}, "R": {"x": 0.75, "y": 0.45}, ...}
```

### `/api/bake` SSE 응답 형식

```
data: {"status": "processing", "message": "Loading StyleGAN3 model...", "progress": 0.0}
data: {"status": "processing", "message": "Generating frames...", "progress": 0.5}
data: {"status": "complete", "video_url": "/api/video/output_xxx.mp4", "progress": 1.0}
data: {"status": "error", "message": "...", "progress": -1}
```

---

## 핵심 알고리즘

### 1. Spatial Energy Map (spatial_modulator.py)
12개 채널의 `(x, y)` 좌표와 RMS 값을 결합하여 2D 가우시안 블롭을 합산.
이 에너지 맵에서 Sobel 기반 기울기를 계산하여 Warp Field를 생성.

```
E(x,y,t) = Σ_ch  RMS(ch,t) × exp(-||(x,y) - (x_ch, y_ch)||² / 2σ²)
```

### 2. StyleGAN3 W-space Modulation (stylegan3_engine.py)
프리트레인된 StyleGAN3 생성기의 W 벡터를 오디오 에너지에 따라 변조.
- **누적 드리프트**: 매 프레임 미세하게 진화 → 유기적 텍스처 변화
- **에너지 점프**: 비트에 맞춰 중간 레이어 W를 강하게 변조 → 극적인 시각 변화
- 모델은 CPU에 상주, 프레임 생성 시에만 MPS로 이동 후 즉시 반환

### 3. 5-Screen Compositing (video_baker.py)
매 프레임:
1. 레퍼런스 이미지를 5개 스크린 비율로 크롭/배치
2. 채널 마커 위치에 RMS 비례 방사형 왜곡 (`scipy.ndimage.map_coordinates`)
3. 채널 색상별 가우시안 glow 추가
4. StyleGAN3 텍스처를 채널 위치에 최대 80% 블렌딩 (채널 색상 틴트)
5. 채널 마커 + 에너지 링 오버레이 (PIL ImageDraw)

### 4. M4 MacBook MPS 메모리 관리
16GB 통합 메모리 제약 하에서 StyleGAN3 512px 모델 구동:
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` (최대 메모리 사용)
- `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7`
- 프레임별 `CPU→MPS→synthesis→CPU→flush` 사이클
- `torch.mps.synchronize()` + `empty_cache()` + `gc.collect()`

---

## 실행 방법

### 사전 요구사항
- Node.js 18+
- Python 3.11+
- FFmpeg (`brew install ffmpeg`)

### 프론트엔드

```bash
npm install
npm run dev
# http://localhost:5173
```

### 백엔드

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7 \
  uvicorn backend.main:app --reload --port 8000
```

StyleGAN3 모델(`stylegan3-t-metfaces-1024x1024.pkl`)은 첫 실행 시 자동 다운로드됩니다.

### 백엔드 없이 실행 (시뮬레이션 모드)
`src/App.jsx`에서 `USE_BACKEND = false`로 설정하면 백엔드 없이 프론트만으로 동작합니다.
Bake 단계는 프로그레스 시뮬레이션으로 대체됩니다.

---

## 7.1.4 채널 배치

```
         Ltf ---- Rtf          (Top Front)
        /    \  /    \
      Ltr --- C --- Rtr        (Top Rear)
     /   \   |   /   \
    L --- Ls-|-Rs --- R        (Ear Level)
     \   /   |   \   /
      Lrs ---+--- Rrs         (Rear)
             |
            LFE               (Subwoofer)
```

| 채널 | 이름 | 기본 위치 |
|------|------|-----------|
| L | Left | (0.25, 0.45) |
| R | Right | (0.75, 0.45) |
| C | Center | (0.50, 0.40) |
| LFE | Subwoofer | (0.50, 0.70) |
| Ls | Left Surround | (0.10, 0.50) |
| Rs | Right Surround | (0.90, 0.50) |
| Lrs | Left Rear Surround | (0.08, 0.75) |
| Rrs | Right Rear Surround | (0.92, 0.75) |
| Ltf | Left Top Front | (0.30, 0.20) |
| Rtf | Right Top Front | (0.70, 0.20) |
| Ltr | Left Top Rear | (0.20, 0.30) |
| Rtr | Right Top Rear | (0.80, 0.30) |

---

## 생성 영상 사양

| 항목 | 값 |
|------|-----|
| 해상도 | 1920 x 1080 (Full HD) |
| FPS | 30 |
| 코덱 | H.264 (libx264), CRF 18 |
| 오디오 | AAC 256kbps, 스테레오 다운믹스 |
| StyleGAN3 텍스처 | 1024 → 512 리사이즈 (MetFaces 유화 모델) |
