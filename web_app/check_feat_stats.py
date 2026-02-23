import torch
import numpy as np
import os
import sys

# Add EchoMimic to path
ECHOMIMIC_DIR = "e:\\Work\\personaplex\\echomimic_v2"
sys.path.append(ECHOMIMIC_DIR)

from src.models.whisper.audio2feature import load_audio_model

def compare_models(audio_path, device="cuda"):
    print(f"Comparing features for {audio_path}...")
    
    # Tiny model path
    whisper_path = os.path.join(ECHOMIMIC_DIR, "pretrained_weights/audio_processor/tiny.pt")
    wav2vec_path = os.path.join(ECHOMIMIC_DIR, "pretrained_weights/wav2vec2-base-960h")
    
    # Load models
    print("Loading Whisper...")
    whisper_proc = load_audio_model(whisper_path, device, model_type="whisper")
    whisper_proc.model = whisper_proc.model.float()
    
    print("Loading Wav2Vec2...")
    wav2vec_proc = load_audio_model(wav2vec_path, device, model_type="wav2vec2", adapter_path=None)
    wav2vec_proc.model = wav2vec_proc.model.float()
    wav2vec_proc.projection = wav2vec_proc.projection.float()
    
    # Extract features
    print("Extracting features...")
    w_feat = whisper_proc.audio2feat(audio_path)
    wv_feat = wav2vec_proc.audio2feat(audio_path)
    
    print("\nWhisper Feature Stats:")
    print(f"  Shape: {w_feat.shape}")
    print(f"  Mean: {np.mean(w_feat):.4f}")
    print(f"  Std:  {np.std(w_feat):.4f}")
    print(f"  Min:  {np.min(w_feat):.4f}")
    print(f"  Max:  {np.max(w_feat):.4f}")
    
    print("\nWav2Vec2 Feature Stats (Pre-Projection):")
    # We want to see the RAW features from Wav2Vec2 (last_hidden_state)
    # But current audio2feat applies the projection (Pseudo-Identity)
    print(f"  Shape: {wv_feat.shape}")
    print(f"  Mean: {np.mean(wv_feat):.4f}")
    print(f"  Std:  {np.std(wv_feat):.4f}")
    print(f"  Min:  {np.min(wv_feat):.4f}")
    print(f"  Max:  {np.max(wv_feat):.4f}")

if __name__ == "__main__":
    # Use a dummy audio if available, or just use the assets
    audio = os.path.join(ECHOMIMIC_DIR, "assets/test.mp3")
    if not os.path.exists(audio):
        # Create a tiny silence if asset not found
        import librosa
        import soundfile as sf
        silence = np.zeros(16000*2)
        sf.write("dummy.wav", silence, 16000)
        audio = "dummy.wav"
        
    compare_models(audio)
