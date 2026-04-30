import sys
print(f"Python version: {sys.version}")

import torch
print(f"PyTorch version: {torch.__version__}")

import librosa
print(f"librosa version: {librosa.__version__}")

import cv2
print(f"OpenCV version: {cv2.__version__}")

import mediapipe as mp
print(f"MediaPipe version: {mp.__version__}")

import yaml
import pandas
import numpy as np
import pydub
print("All Python packages: OK")

import subprocess
result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
if result.returncode == 0:
    print(f"FFmpeg: OK ({result.stdout.split('\\n')[0]})")
else:
    print("FFmpeg: NOT FOUND — install it before proceeding")

import os
pann_checkpoint = "models/panns/Cnn14_mAP=0.431.pth"
if os.path.exists(pann_checkpoint):
    size_mb = os.path.getsize(pann_checkpoint) / (1024 * 1024)
    print(f"PANNs checkpoint: OK ({size_mb:.1f} MB)")
else:
    print("PANNs checkpoint: NOT FOUND — run the wget command above")

sample_video = "data/input/sample_hindi.mp4"
if os.path.exists(sample_video):
    size_mb = os.path.getsize(sample_video) / (1024 * 1024)
    print(f"Sample video: OK ({size_mb:.1f} MB)")
else:
    print("Sample video: NOT FOUND — download a Hindi video")

print("\n✓ Environment verification complete.")
