import sys
import os
import numpy as np

# add path
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "echomimic_v2"))
from src.models.whisper.audio2feature import load_audio_model
from src.models.whisper.whisper.audio import load_audio

def main():
    audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "echomimic_v2", "assets", "halfbody_demo", "audio", "chinese", "echomimicv2_woman.wav")
    
    # Load model
    print("Loading model...")
    audio_processor = load_audio_model(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "echomimic_v2", "pretrained_weights", "audio_processor", "tiny.pt"), device="cuda", model_type="whisper")
    
    # 1. Path input
    feat_path = audio_processor.audio2feat(audio_path)
    print("Feature from path shape:", feat_path.shape)
    print("Feature from path [0, :5]:", feat_path[0, :5])
    
    # 2. Numpy array input
    wav_np = load_audio(audio_path)
    feat_np = audio_processor.audio2feat(wav_np)
    print("Feature from numpy shape:", feat_np.shape)
    print("Feature from numpy [0, :5]:", feat_np[0, :5])
    
    # Compare
    diff = np.abs(feat_path - feat_np).max()
    print("Max absolute difference:", diff)

    # 3. Short numpy array (140 clip frames -> ~5.833s)
    short_np = wav_np[:int(16000 * 140 / 24)]
    feat_short = audio_processor.audio2feat(short_np)
    print("Feature from 140-frame numpy shape:", feat_short.shape)
    
    # Check max abs diff for the first few frames
    min_len = min(feat_path.shape[0], feat_short.shape[0])
    diff_140 = np.abs(feat_path[:min_len] - feat_short).max()
    print("Max absolute difference (140 frames vs full):", diff_140)

if __name__ == "__main__":
    main()
