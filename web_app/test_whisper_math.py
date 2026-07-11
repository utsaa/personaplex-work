import sys
import numpy as np
import torch
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "echomimic_v2", "src", "models", "whisper"))
import whisper
model = whisper.load_model("tiny", device="cuda")
# create 2.5 seconds of fake audio
audio = np.random.randn(16000 * 3).astype(np.float32)
res = model.transcribe(audio)
print("Segments:", len(res['segments']))
for emb in res['segments']:
    print("Start:", emb['start'], "End:", emb['end'])
    enc = emb['encoder_embeddings']
    print("Encoder shape:", enc.shape)
