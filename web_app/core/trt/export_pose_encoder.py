import torch
import torch.nn as nn
from src.models.pose_encoder import PoseEncoder
import os
import sys

def ensure_paths(echomimic_dir):
    if echomimic_dir not in sys.path:
        sys.path.insert(0, echomimic_dir)

def export_pose_encoder_to_onnx(pt_path, onnx_path, echomimic_dir, device="cuda"):
    ensure_paths(echomimic_dir)
    
    # Initialize model with same params as core/models.py
    model = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(device=device, dtype=torch.float16)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()

    # Dummy input: (B, C, F, H, W)
    dummy_input = torch.randn(1, 3, 12, 512, 512, dtype=torch.float16, device=device)
    
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['conditioning'],
        output_names=['embedding'],
        dynamic_axes={
            'conditioning': {0: 'batch', 2: 'frames'},
            'embedding': {0: 'batch', 2: 'frames'}
        },
        opset_version=17
    )
    print(f"[TRT] Pose Encoder ONNX export complete: {onnx_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-path", type=str, required=True)
    parser.add_argument("--onnx-path", type=str, required=True)
    parser.add_argument("--echomimic-dir", type=str, required=True)
    args = parser.parse_args()
    export_pose_encoder_to_onnx(args.pt_path, args.onnx_path, args.echomimic_dir)
