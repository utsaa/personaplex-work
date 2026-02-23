import torch
import os

def find_layers(path):
    print(f"Opening {path}...")
    try:
        weights = torch.load(path, map_location="cpu")
        print(f"Total keys: {len(weights.keys())}")
        
        found = []
        for key, val in weights.items():
            if isinstance(val, torch.Tensor):
                shape = list(val.shape)
                # We are looking for something that takes 768 as input (dim 1 in linear weight)
                # and projects to something (dim 0)
                if len(shape) == 2 and shape[1] == 768:
                    found.append((key, shape))
                if "audio" in key.lower() or "proj" in key.lower():
                    # Just in case, list anything with these keywords
                    if len(shape) <= 2:
                        found.append((key, shape))
        
        if found:
            print("\nPotential Projection Layers Found:")
            for k, s in found:
                print(f"  {k} -> {s}")
        else:
            print("\nNo obvious projection layers found. Here are first 20 keys:")
            for k in list(weights.keys())[:20]:
                print(f"  {k}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_layers("../echomimic_v2/pretrained_weights/audio_processor/musetalk/unet.pth")
