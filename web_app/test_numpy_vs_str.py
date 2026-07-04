import sys
import numpy as np
import torch
import soundfile as sf
sys.path.append("/workspace/personaplex-work/echomimic_v2")
from src.models.whisper.audio2feature import Audio2Feature

device = "cuda" if torch.cuda.is_available() else "cpu"
audio_guider = Audio2Feature(model_path="/workspace/personaplex-work/echomimic_v2/pretrained_weights/audio_processor/tiny.pt", device=device)

wav_path = "/workspace/personaplex-work/echomimic_v2/assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"
audio_np, sr = sf.read(wav_path)
audio_np = audio_np.mean(axis=1) # Convert to mono
audio_np = audio_np.astype(np.float32)

feat_str = audio_guider.audio2feat(wav_path)
print("STRING SUM:", np.sum(feat_str, dtype=np.float32))

feat_np = audio_guider.audio2feat(audio_np)
print("NUMPY SUM:", np.sum(feat_np, dtype=np.float32))

if np.array_equal(feat_str, feat_np):
    print("THE FEATURES ARE IDENTICAL!")
else:
    print("THE FEATURES ARE DIFFERENT!")
