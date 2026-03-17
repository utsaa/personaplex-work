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
from typing import Optional


def patch_transformer_for_trt(model, video_length: int = 13):
    """
    Patches all transformer blocks in the UNet to handle a 'reference' frame
    passed as the first frame of the input.

    NOTE: This is NOT called during ONNX export — the standard forward traces correctly.
    This is intended for torch.compile / other acceleration paths where you need
    to override the attention blocks with a static-shape-compatible version.

    Args:
        video_length: Total frames including the reference frame (clip_frames + 1).
                      Must be set to the actual value, not left as default if clip_frames != 12.
    """
    def make_forward(vl: int):
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
            # Frame 0 is the reference frame.
            _vl = vl if video_length is None else video_length

            b_f = hidden_states.shape[0]
            f = _vl
            b = b_f // f

            # 1. Normalization
            norm_hidden_states = self.norm1(hidden_states)

            # 2. Extract Reference Features (Frame 0)
            l = norm_hidden_states.shape[1]
            c = norm_hidden_states.shape[2]
            norm_hidden_states_reshaped = norm_hidden_states.view(b, f, l, c)
            ref_feat = norm_hidden_states_reshaped[:, 0:1, :, :]  # (B, 1, L, C)

            # Repeat reference for all frames, then concatenate for mutual attention
            ref_feat_expanded = ref_feat.expand(b, f, l, c).reshape(b_f, l, c)
            modify_norm_hidden_states = torch.cat([norm_hidden_states, ref_feat_expanded], dim=1)

            # 3. Self-Attention (Mutual)
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=modify_norm_hidden_states,
                attention_mask=None,
            )
            hidden_states = attn_output + hidden_states

            # 4. Audio Cross-Attention (Attn2) — only if block has it
            if hasattr(self, 'attn2') and self.attn2 is not None and audio_cond_fea is not None:
                norm_hidden_states = self.norm2(hidden_states)
                hidden_states = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=audio_cond_fea,
                ) * 3.0 + hidden_states

            # 5. Feed-forward
            hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
            return hidden_states
        return new_forward

    patched_forward = make_forward(video_length)
    for m in model.modules():
        if isinstance(m, (BasicTransformerBlock, TemporalBasicTransformerBlock)):
            m.forward = patched_forward.__get__(m, m.__class__)
    return model

def export_unet_to_onnx(pt_path, onnx_path, echomimic_dir, base_model_path=None, quantize_fp8=False, clip_frames=12, height=512, width=512, use_safetensors=False):
    ensure_paths(echomimic_dir)

    # Load config to get model parameters
    config_path = os.path.join(echomimic_dir, "configs", "inference", "inference_v2.yaml")
    infer_config = OmegaConf.load(config_path)

    print(f"[TRT] Loading UNet for export: {pt_path}")

    # If base_model_path is not provided, try to infer it from pt_path (legacy behavior)
    if base_model_path is None:
        base_model_path = pt_path.split('denoising_unet')[0]

    model = EMOUNet3DConditionModel.from_pretrained_2d(
        base_model_path,
        "",  # No motion module path — handled separately in EchoMimic
        subfolder="unet" if os.path.isdir(pt_path.split('denoising_unet')[0]) else None,
        unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs),
        use_safetensors=use_safetensors,
    ).to(dtype=torch.float16, device="cpu")

    # Load the actual weights to CPU
    print(f"[TRT] Loading weights {pt_path} to CPU...")
    state_dict = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    del state_dict  # Free CPU RAM ASAP
    torch.cuda.empty_cache()

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
    # Patch the UNet for TRT-compatible Mutual Attention and Audio Cross-Attention
    print(f"[TRT] Patching UNet transformer blocks for TRT (video_length={clip_frames+1})...")
    patch_transformer_for_trt(model, video_length=clip_frames+1)

    model.eval()

    # Dummy input dimensions — must match the trace shapes TRT will use at runtime.
    # F = clip_frames + 1 reference frame.
    F = clip_frames + 1
    B = 2   # CFG batch
    H, W = height // 8, width // 8  # latent spatial size

    dummy_sample = torch.randn(B, 4, F, H, W, dtype=torch.float16, device="cpu")
    dummy_timestep = torch.tensor([1.0] * B, dtype=torch.float16, device="cpu")
    # face_musk_fea = PoseEncoder output: (B, 320, F, H, W).
    dummy_pose = torch.randn(B, 320, F, H, W, dtype=torch.float16, device="cpu")

    # ------------------------------------------------------------------ #
    # Wrap the UNet to expose ONLY the 4 real inputs we care about.       #
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

    wrapped = UNetExportWrapper(model).eval().cpu()
    torch.cuda.empty_cache()

    # audio_cond_fea: pipeline sends (B, F, n_audio, cross_attention_dim) 4D.
    # cross_attention_dim=384 from inference_v2.yaml (matches to_k input dim).
    # transformer_3d.py rearranges (B,F,n,384) → (B*F, n, 384) before passing to blocks.
    AUDIO_DIM = 384  # cross_attention_dim
    # Audio (B, F, seq_len, 384). Typical seq_len is 50 (margin=2) or 130 (margin=6).
    dummy_audio = torch.randn(B, F, 130, AUDIO_DIM, dtype=torch.float16, device="cpu")


    print(f"[TRT] Clean VRAM before export: {torch.cuda.memory_allocated()/1024**2:.2f}MB allocated")
    print(f"[TRT] Starting CPU-based ONNX export...")
    print(f"[TRT] Exporting to ONNX: {onnx_path}")
    try:
        # Export directly to onnx_path as a string file path.
        # PyTorch's C++ backend (graph._export_onnx) sets model_file_location = f when
        # f is a str, which lets the C++ code write model weights as external data
        # (onnx_path + ".data") alongside the graph file when the model exceeds 2GB.
        # This is the ONLY supported path for >2GB models — do NOT pass BytesIO.
        torch.onnx.export(
            wrapped,
            (dummy_sample, dummy_timestep, dummy_audio, dummy_pose),
            onnx_path,          # Must be a string path — not BytesIO
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['sample', 'timestep', 'audio_cond_fea', 'face_musk_fea'],
            output_names=['out_sample'],
            dynamic_axes={
                'sample': {0: 'batch', 2: 'frames'},
                'timestep': {0: 'batch'},
                'audio_cond_fea': {0: 'batch', 1: 'frames', 2: 'seq_len'},
                'face_musk_fea': {0: 'batch', 2: 'frames'},
                'out_sample': {0: 'batch', 2: 'frames'}
            }
        )
        # PyTorch has written onnx_path (graph) + onnx_path + ".data" (weights).
        # TensorRT's parse_from_file(onnx_path) will find both files automatically.
    except Exception as e:
        import traceback
        print(f"[TRT] ONNX export failed: {e}")
        traceback.print_exc()
        raise
    print(f"[TRT] ONNX export complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-path", type=str, required=True)
    parser.add_argument("--onnx-path", type=str, required=True)
    parser.add_argument("--echomimic-dir", type=str, required=True)
    parser.add_argument("--clip-frames", type=int, default=12)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--base-model-path", type=str, default=None)
    parser.add_argument("--use-safetensors", action="store_true", help="Use safetensors for loading")
    args = parser.parse_args()
    export_unet_to_onnx(args.pt_path, args.onnx_path, args.echomimic_dir, base_model_path=args.base_model_path, clip_frames=args.clip_frames, height=args.height, width=args.width, use_safetensors=args.use_safetensors)
