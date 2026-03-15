import torch
import torch.nn as nn
from diffusers import AutoencoderKL
import os
import sys

def ensure_paths(echomimic_dir):
    if echomimic_dir not in sys.path:
        sys.path.insert(0, echomimic_dir)

class VAEEncoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    def forward(self, x):
        return self.vae.encode(x).latent_dist.mean

class VAEDecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    def forward(self, z):
        return self.vae.decode(z).sample

def export_vae_to_onnx(model_path, onnx_dir, device="cuda", height=512, width=512):
    vae = AutoencoderKL.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    vae.eval()

    # 1. Encoder
    encoder = VAEEncoderWrapper(vae)
    dummy_img = torch.randn(1, 3, height, width, dtype=torch.float16, device=device)
    encoder_onnx = os.path.join(onnx_dir, f"vae_encoder_{width}x{height}.onnx")
    torch.onnx.export(
        encoder, dummy_img, encoder_onnx,
        input_names=['input'], output_names=['latent'],
        dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'}, 'latent': {0: 'batch', 2: 'h', 3: 'w'}},
        opset_version=17
    )

    # 2. Decoder
    decoder = VAEDecoderWrapper(vae)
    h_lat, w_lat = height // 8, width // 8
    dummy_latent = torch.randn(1, 4, h_lat, w_lat, dtype=torch.float16, device=device)
    decoder_onnx = os.path.join(onnx_dir, f"vae_decoder_{width}x{height}.onnx")
    torch.onnx.export(
        decoder, dummy_latent, decoder_onnx,
        input_names=['latent'], output_names=['sample'],
        dynamic_axes={'latent': {0: 'batch', 2: 'h', 3: 'w'}, 'sample': {0: 'batch', 2: 'height', 3: 'width'}},
        opset_version=17
    )
    print(f"[TRT] VAE ONNX export complete: {onnx_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--onnx-dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.onnx_dir, exist_ok=True)
    export_vae_to_onnx(args.model_path, args.onnx_dir)
