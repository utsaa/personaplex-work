import torch
import torch.nn as nn
import os
import sys

def ensure_paths(echomimic_dir):
    if echomimic_dir not in sys.path:
        sys.path.insert(0, echomimic_dir)

def export_ref_unet_to_onnx(model_path, onnx_path, echomimic_dir, base_model_path=None, device="cuda", height=512, width=512):
    ensure_paths(echomimic_dir)
    from src.models.unet_2d_condition import UNet2DConditionModel
    # Reference UNet is 2D
    # It might be in a subfolder or a .pth file.
    # EchoMimic loads it from .pth but initializes from pretrained_base_model_path.
    
    base_path = base_model_path
    if not base_path or not os.path.exists(base_path):
        # Fallback heuristic if not provided
        base_path = os.path.join(echomimic_dir, "pretrained_weights", "sd-image-variations-diffusers")
        if not os.path.isdir(base_path):
             base_path = "./pretrained_weights/sd-image-variations-diffusers"

    unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=torch.float16).to(device)
    
    # Load weights
    print(f"[TRT] Loading weights {model_path} to {device}...")
    state_dict = torch.load(model_path, map_location=device)
    # If the state_dict is nested (e.g. from some training scripts), unwrap it
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # Handle possible "unet." prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("unet."):
            new_state_dict[k[5:]] = v
        else:
            new_state_dict[k] = v
            
    unet.load_state_dict(new_state_dict, strict=False)
    del state_dict
    del new_state_dict
    torch.cuda.empty_cache()
    unet.eval()

    h_lat, w_lat = height // 8, width // 8
    dummy_latent = torch.randn(1, 4, h_lat, w_lat, dtype=torch.float16, device=device)
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
    parser.add_argument("--echomimic-dir", type=str, required=True)
    parser.add_argument("--base-model-path", type=str, default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()
    export_ref_unet_to_onnx(args.pt_path, args.onnx_path, args.echomimic_dir, base_model_path=args.base_model_path, height=args.height, width=args.width)
