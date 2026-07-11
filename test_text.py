import sys
import torch
import librosa
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "echomimic_v2"))
import whisper
model = whisper.load_model("base")
result = model.transcribe(os.path.join(os.path.dirname(os.path.abspath(__file__)), "echomimic_v2", "assets", "halfbody_demo", "audio", "chinese", "echomimicv2_woman.wav"))
print("Transcribed Text:")
for seg in result["segments"]:
    print(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}")
