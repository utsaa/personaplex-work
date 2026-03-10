import torch
import torch.nn as nn
from omegaconf import OmegaConf
import os
import sys

# Add echomimic_v2 to path if not already there
def ensure_paths(echomimic_dir):
    if echomimic_dir not in sys.path:
        sys.path.insert(0, echomimic_dir)

from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.attention import BasicTransformerBlock, TemporalBasicTransformerBlock
from einops import rearrange
from typing import Optional, Dict, Any

def patch_transformer_for_trt(model):
    """
    Patches all transformer blocks in the UNet to handle a 'reference' frame
    passed as the first frame of the input.
    """
    def new_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        audio_cond_fea: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        video_length=None,
    ):
        # hidden_states: (B*F, L, C)
        # We assume Frame 0 is the reference frame.
        if video_length is None:
            # For ONNX export, we'll provide a dummy video_length
            video_length = 1 
            
        b_f = hidden_states.shape[0]
        # Calculate B (batch) and F (frames)
        # In TRT export, F is video_length
        f = video_length
        b = b_f // f
        
        # 1. Normalization
        norm_hidden_states = self.norm1(hidden_states)
        
        # 2. Extract Reference Features (Frame 0)
        # Reshape to (B, F, L, C) then slice
        l = norm_hidden_states.shape[1]
        c = norm_hidden_states.shape[2]
        norm_hidden_states_reshaped = norm_hidden_states.view(b, f, l, c)
        ref_feat = norm_hidden_states_reshaped[:, 0:1, :, :] # (B, 1, L, C)
        
        # We only want to process the denoise frames (1..F-1)
        # BUT for ONNX, it's easier to process all and just ignore the output of Frame 0.
        # Actually, let's process all frames to keep the shapes consistent.
        
        # Repeat reference for all frames in the video
        # (B, F, L, C)
        ref_feat_expanded = ref_feat.expand(b, f, l, c).reshape(b_f, l, c)
        
        # Concatenate current features with reference features
        # (B*F, 2*L, C)
        modify_norm_hidden_states = torch.cat([norm_hidden_states, ref_feat_expanded], dim=1)
        
        # 3. Self-Attention (Mutual)
        # Note: We use cross-attention mechanism for mutual-self-attention
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=modify_norm_hidden_states,
            attention_mask=None,
        )
        
        hidden_states = attn_output + hidden_states
        
        # 4. Audio Cross-Attention (Attn2)
        if self.attn2 is not None and audio_cond_fea is not None:
            norm_hidden_states = self.norm2(hidden_states)
            hidden_states = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=audio_cond_fea,
            ) * 3.0 + hidden_states
            
        # 5. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        
        return hidden_states

    for m in model.modules():
        if isinstance(m, (BasicTransformerBlock, TemporalBasicTransformerBlock)):
            # Store original if needed, but we don't for TRT
            m.forward = new_forward.__get__(m, m.__class__)
    return model

def export_unet_to_onnx(pt_path, onnx_path, echomimic_dir, base_model_path=None, quantize_fp8=False):
    ensure_paths(echomimic_dir)
    
    # Load config to get model parameters
    config_path = os.path.join(echomimic_dir, "configs", "inference", "inference_v2.yaml")
    infer_config = OmegaConf.load(config_path)
    
    print(f"[TRT] Loading UNet for export: {pt_path}")
    # Instantiate with dummy motion path first to avoid loading two weights
    
    # If base_model_path is not provided, try to infer it from pt_path (legacy behavior)
    if base_model_path is None:
        base_model_path = pt_path.split('denoising_unet')[0]

    model = EMOUNet3DConditionModel.from_pretrained_2d(
        base_model_path, # folder containing unet
        "", # No motion module path here as it's separate in EchoMimic
        subfolder="unet" if os.path.isdir(pt_path.split('denoising_unet')[0]) else None,
        unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs),
    ).to(dtype=torch.float16, device="cuda")
    
    # Load the actual weights
    state_dict = torch.load(pt_path, map_location="cuda")
    model.load_state_dict(state_dict, strict=False)
    
    # FP8 Quantization (torchao)
    if quantize_fp8:
        try:
            from torchao.quantization import quantize_, Float8WeightOnlyConfig
            print(f"[TRT] Quantizing Denoising UNet weights to FP8 before export...")
            quantize_(model, Float8WeightOnlyConfig())
        except ImportError:
            try:
                from torchao.quantization import quantize_, float8_weight_only
                print(f"[TRT] Using float8_weight_only alias for FP8 export...")
                quantize_(model, float8_weight_only())
            except ImportError:
                print("[TRT] [ERROR] torchao not found. Skipping FP8 quantization.")
        except Exception as e:
            print(f"[TRT] [ERROR] FP8 Quantization failed: {e}")

    # Patch for TRT
    model = patch_transformer_for_trt(model)
    model.eval()
    # Dummy input dimensions
    F = 13  # clip_frames(12) + 1 reference
    B = 2   # CFG batch
    H, W = 64, 64  # latent spatial size

    dummy_sample = torch.randn(B, 4, F, H, W, dtype=torch.float16, device="cuda")
    dummy_timestep = torch.tensor([1.0] * B, dtype=torch.float16, device="cuda")
    # face_musk_fea = PoseEncoder output: (B, 320, F, H, W).
    # PoseEncoder(320, ...) outputs 320 channels; 3×stride-2 on 512px → 64px = latent size.
    # The UNet does `sample = conv_in(sample) + face_musk_fea` so shapes must match.
    dummy_pose = torch.randn(B, 320, F, H, W, dtype=torch.float16, device="cuda")

    # ------------------------------------------------------------------ #
    # Wrap the UNet to expose ONLY the 4 real inputs we care about.       #
    # The UNet forward has many optional keyword args                      #
    # (encoder_hidden_states, class_labels, attention_mask, …).          #
    # Passing them as positional args is fragile and was the root cause.  #
    # ------------------------------------------------------------------ #
    class UNetExportWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, audio_cond_fea, face_musk_fea):
            out = self.unet(
                sample,
                timestep,
                encoder_hidden_states=None,
                audio_cond_fea=audio_cond_fea,
                face_musk_fea=face_musk_fea,
                return_dict=False,
            )
            return out[0]  # (sample,) tuple → just the tensor

    wrapped = UNetExportWrapper(model).eval()

    # audio_cond_fea: pipeline sends (B, F, n_audio, cross_attention_dim) 4D.
    # cross_attention_dim=384 from inference_v2.yaml (matches to_k input dim).
    # transformer_3d.py rearranges (B,F,n,384) → (B*F, n, 384) before passing to blocks.
    AUDIO_DIM = 384  # cross_attention_dim
    dummy_audio = torch.randn(B, F, 1, AUDIO_DIM, dtype=torch.float16, device="cuda")

    print(f"[TRT] Exporting to ONNX: {onnx_path}")
    try:
        torch.onnx.export(
            wrapped,
            (dummy_sample, dummy_timestep, dummy_audio, dummy_pose),
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['sample', 'timestep', 'audio_cond_fea', 'face_musk_fea'],
            output_names=['out_sample'],
            dynamic_axes={
                'sample': {0: 'batch', 2: 'frames'},
                'timestep': {0: 'batch'},
                'audio_cond_fea': {0: 'batch', 1: 'frames'},  # (B, F, n_audio, 1280)
                'face_musk_fea': {0: 'batch', 2: 'frames'},
                'out_sample': {0: 'batch', 2: 'frames'}
            }
        )
    except Exception as e:
        import traceback
        print(f"[TRT] ONNX export failed: {e}")
        traceback.print_exc()
        raise
    print(f"[TRT] ONNX export complete.")

if __name__ == "__main__":
    # Test script run
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-path", type=str, required=True)
    parser.add_argument("--onnx-path", type=str, required=True)
    parser.add_argument("--echomimic-dir", type=str, required=True)
    args = parser.parse_args()
    export_unet_to_onnx(args.pt_path, args.onnx_path, args.echomimic_dir)
