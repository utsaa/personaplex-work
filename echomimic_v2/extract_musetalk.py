import torch
import os
import sys

def verify_and_extract(musetalk_path, output_path):
    if not os.path.exists(musetalk_path):
        print(f"Error: MuseTalk weights not found at {musetalk_path}")
        print("Please download pytorch_model.bin from https://huggingface.co/TMElyralab/MuseTalk/tree/main")
        return

    print(f"Loading MuseTalk weights from {musetalk_path}...")
    weights = torch.load(musetalk_path, map_location="cpu")

    print("--- Searching for Audio Projection Keys ---")
    adapter_weights = {}
    found_keys = []
    
    # Common keys in MuseTalk: 'audio_projection.weight', 'mu_audio_proj.weight', 'audio_linear.weight'
    target_patterns = ["audio_projection", "mu_audio_proj", "audio_linear", "audio_proj"]
    
    for key in weights.keys():
        if any(pattern in key.lower() for pattern in target_patterns):
            print(f"FOUND KEY: {key} | Shape: {weights[key].shape}")
            adapter_weights[key] = weights[key]
            found_keys.append(key)

    if not found_keys:
        print("No audio projection keys found! Printing all keys for debugging:")
        for k in sorted(list(weights.keys())):
            print(k)
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(adapter_weights, output_path)
    print(f"\nSUCCESS: Extracted {len(found_keys)} keys to {output_path}")

if __name__ == "__main__":
    # Default paths
    src = "./pretrained_weights/audio_processor/musetalk/unet.pth"
    dst = "./pretrained_weights/audio_processor/wav2vec2/trained_adapter.pt"
    
    if len(sys.argv) > 1:
        src = sys.argv[1]
    if len(sys.argv) > 2:
        dst = sys.argv[2]
        
    verify_and_extract(src, dst)
