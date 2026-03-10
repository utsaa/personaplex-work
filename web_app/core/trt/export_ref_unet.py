import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
import os
import sys

def export_ref_unet_to_onnx(model_path, onnx_path, device="cuda"):
    # Reference UNet is 2D
    # It might be in a subfolder or a .pth file.
    # EchoMimic loads it from .pth but initializes from pretrained_base_model_path.
    
    # We'll use the same logic as core/models.py
    # reference_unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet")
    # reference_unet.load_state_dict(torch.load(pth_path))
    
    # For now, let's assume pt_path is the .pth file and we need the base_path
    base_path = model_path.split('reference_unet.pth')[0] # Heuristic
    if not os.path.isdir(base_path):
        # Fallback if manual pathing fails
        base_path = "./pretrained_weights/sd-image-variations-diffusers"

    unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=torch.float16).to(device)
    unet.load_state_dict(torch.load(model_path, map_location=device))
    unet.eval()

    dummy_latent = torch.randn(1, 4, 64, 64, dtype=torch.float16, device=device)
    dummy_t = torch.tensor([1.0], dtype=torch.float16, device=device)
    dummy_cond = torch.randn(1, 1, 768, dtype=torch.float16, device=device) # SD Variations cond
    
    torch.onnx.export(
        unet, (dummy_latent, dummy_t, dummy_cond), onnx_path,
        input_names=['sample', 'timestep', 'encoder_hidden_states'],
        output_names=['out_sample'],
        dynamic_axes={
            'sample': {0: 'batch', 2: 'height', 3: 'width'},
            'timestep': {0: 'batch'},
            'encoder_hidden_states': {0: 'batch'},
            'out_sample': {0: 'batch', 2: 'height', 3: 'width'}
        },
        opset_version=17
    )
    print(f"[TRT] Reference UNet ONNX export complete: {onnx_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-path", type=str, required=True)
    parser.add_argument("--onnx-path", type=str, required=True)
    args = parser.parse_args()
    export_ref_unet_to_onnx(args.pt_path, args.onnx_path)
