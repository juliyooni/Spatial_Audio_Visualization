import sounddevice as sd
import soundfile as sf
import pandas as pd
import numpy as np
from datetime import datetime

# --- 설정 ---
CHANNELS = 12  # 7.1.4 채널 (BlackHole 1~12번 사용)
FS = 44100  # 샘플링 레이트
DEVICE_NAME = "BlackHole 16ch"

# 저장용 리스트 및 배열
raw_audio_buffer = []
rms_data_list = []

print(f"--- 7.1.4 채널 데이터 캡처 시작 ---")
print(f"출력 기기가 '{DEVICE_NAME}'으로 설정되어 있는지 확인하세요.")


def callback(indata, frames, time, status):
    if status:
        print(status)

    # 1. 1~12번 채널 데이터만 추출 (7.1.4)
    target_channels = indata[:, :CHANNELS].copy()

    # 2. WAV 파일용 원본 오디오 데이터 저장
    raw_audio_buffer.append(target_channels)

    # 3. CSV 파일용 실시간 에너지(RMS) 계산
    # 각 채널별로 제곱근 평균 제곱(RMS)을 구해 수치화합니다.
    rms = np.sqrt(np.mean(target_channels**2, axis=0))
    rms_data_list.append(rms)

    # 진행 상황 출력 (한 줄로 업데이트)
    print(f"\r녹음 중... (에너지: {np.mean(rms):.4f})", end="")


# --- 실행 ---
try:
    # BlackHole은 16채널 장치이므로 channels=16으로 열어야 에러가 안 납니다.
    with sd.InputStream(
        device=DEVICE_NAME, channels=16, samplerate=FS, callback=callback
    ):
        print("\n녹음을 시작합니다. 중단하려면 Ctrl+C를 누르세요.")
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\n\n녹음 중단. 파일 저장 중...")

    # 파일명 생성 (현재 시간 기준)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = f"audio_source_{timestamp}.wav"
    csv_filename = f"audio_features_{timestamp}.csv"

    # 1. 원본 오디오(WAV) 저장
    all_audio = np.concatenate(raw_audio_buffer, axis=0)
    sf.write(wav_filename, all_audio, FS)
    print(f"✅ 멀티채널 음원 저장 완료: {wav_filename}")

    # 2. 에너지 데이터(CSV) 저장
    columns = [
        "L",
        "R",
        "C",
        "LFE",
        "Ls",
        "Rs",
        "Lrs",
        "Rrs",
        "Ltf",
        "Rtf",
        "Ltr",
        "Rtr",
    ]
    df = pd.DataFrame(rms_data_list, columns=columns)
    df.to_csv(csv_filename, index=False)
    print(f"✅ AI 제어용 CSV 데이터 저장 완료: {csv_filename}")

except Exception as e:
    print(f"\n에러 발생: {e}")
