import sys
import numpy as np
import torch
sys.path.append("/workspace/personaplex-work/echomimic_v2")
from src.models.whisper.audio2feature import Audio2Feature
from src.models.whisper.whisper.audio import load_audio

device = "cuda" if torch.cuda.is_available() else "cpu"
audio_guider = Audio2Feature(model_path="/workspace/personaplex-work/echomimic_v2/pretrained_weights/audio_processor/tiny.pt", device=device)

wav_path = "/workspace/personaplex-work/echomimic_v2/assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"
audio_np = load_audio(wav_path)[:16000 * 3] # 3.0s of speech

silence = np.zeros(16000 * 2, dtype=np.float32) # 2.0s of silence
padded_audio = np.concatenate((silence, audio_np))

feat_np = audio_guider.audio2feat(audio_np)
feat_padded = audio_guider.audio2feat(padded_audio)
# Shift the padded features by 2.0s (100 frames) to align with the non-padded ones
feat_padded_shifted = feat_padded[100:]

print("Original shape:", feat_np.shape)
print("Padded shifted shape:", feat_padded_shifted.shape)

min_len = min(len(feat_np), len(feat_padded_shifted))
feat_np = feat_np[:min_len]
feat_padded_shifted = feat_padded_shifted[:min_len]

diff = np.abs(feat_np - feat_padded_shifted).mean()
print("Mean absolute difference:", diff)
print("Original max:", np.max(np.abs(feat_np)))
if diff > 0.1:
    print("FEATURES ARE COMPLETELY DIFFERENT DUE TO SILENCE PADDING!")
else:
    print("FEATURES ARE ALMOST IDENTICAL.")

