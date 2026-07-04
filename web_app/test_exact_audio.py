import sys
import numpy as np
import torch
sys.path.append("/workspace/personaplex-work/echomimic_v2")
from src.models.whisper.audio2feature import Audio2Feature
from src.models.whisper.whisper.audio import load_audio

device = "cuda" if torch.cuda.is_available() else "cpu"
audio_guider = Audio2Feature(model_path="/workspace/personaplex-work/echomimic_v2/pretrained_weights/audio_processor/tiny.pt", device=device)

wav_path = "/workspace/personaplex-work/echomimic_v2/assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"

audio_np = load_audio(wav_path)

feat_str = audio_guider.audio2feat(wav_path)
print("STRING SUM:", np.sum(feat_str, dtype=np.float32))

feat_np = audio_guider.audio2feat(audio_np)
print("NUMPY SUM:", np.sum(feat_np, dtype=np.float32))

if np.array_equal(feat_str, feat_np):
    print("THE FEATURES ARE IDENTICAL!")
else:
    print("THE FEATURES ARE DIFFERENT!")
