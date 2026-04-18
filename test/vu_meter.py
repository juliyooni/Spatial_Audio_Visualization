import sounddevice as sd
import numpy as np
import os

# 설정
DEVICE = "BlackHole 16ch"
CHANNELS = 12

def callback(indata, frames, time, status):
    os.system('clear') # 터미널 화면 청소
    print("--- 12채널 실시간 모니터링 (7.1.4) ---")
    print("음악을 틀었을 때 아래 막대기들이 각각 움직여야 합니다.\n")
    
    # 각 채널별 에너지 계산
    rms = np.sqrt(np.mean(indata[:, :CHANNELS]**2, axis=0))
    
    labels = ['L', 'R', 'C', 'LFE', 'Ls', 'Rs', 'Lrs', 'Rrs', 'Ltf', 'Rtf', 'Ltr', 'Rtr']
    for i, (label, val) in enumerate(zip(labels, rms)):
        # 에너지를 시각적으로 막대기로 표현 (0.1 기준)
        bar = '█' * int(val * 200) 
        print(f"{label.ljust(4)} | {bar}")

try:
    with sd.InputStream(device=DEVICE, channels=16, callback=callback):
        sd.sleep(1000000)
except KeyboardInterrupt:
    print("\n모니터링 종료")