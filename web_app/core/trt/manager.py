import os
import torch
import hashlib
from .builder import TRTEngineBuilder, build_unet_engine
from .runtime import TRTModel

class TRTEngineManager:
    def __init__(self, echomimic_dir, cache_dir="engines"):
        self.echomimic_dir = echomimic_dir
        self.cache_base = os.path.join(echomimic_dir, cache_dir)
        self.gpu_name = torch.cuda.get_device_name(0).replace(" ", "_").lower()
        self.gpu_cache = os.path.join(self.cache_base, self.gpu_name)
        os.makedirs(self.gpu_cache, exist_ok=True)

    def get_unet_engine(self, unet_pt_path, base_model_path=None, force_rebuild=False, fp8=False, clip_frames=12, video_length=25):
        """Gets the path to a compiled UNet engine.
        
        Args:
            clip_frames: Number of audio clip frames (excl. reference). Matches fps * audio_margin.
            video_length: Maximum video length in frames.
        """
        return self._get_engine("denoising_unet_acc", unet_pt_path, force_rebuild,
                               extra_args={"base_model_path": base_model_path, "fp8": fp8,
                                           "clip_frames": clip_frames, "video_length": video_length})

    def get_vae_encoder_engine(self, vae_path, force_rebuild=False):
        return self._get_engine("vae_encoder", vae_path, force_rebuild)

    def get_vae_decoder_engine(self, vae_path, force_rebuild=False):
        return self._get_engine("vae_decoder", vae_path, force_rebuild)

    def get_ref_unet_engine(self, ref_unet_pt_path, force_rebuild=False):
        return self._get_engine("reference_unet", ref_unet_pt_path, force_rebuild)

    def get_pose_encoder_engine(self, pose_pt_path, force_rebuild=False):
        return self._get_engine("pose_encoder", pose_pt_path, force_rebuild)

    def _get_engine(self, model_name, pt_path, force_rebuild, extra_args=None):
        extra_args = extra_args or {}
        fp8 = extra_args.get("fp8", False)
        
        # Suffix engine path if FP8 or specific clip_frames are used
        engine_name = model_name
        if fp8:
            engine_name += "_fp8"
        if model_name == "denoising_unet_acc":
            clip_frames = extra_args.get("clip_frames", 12)
            height = extra_args.get("height", 512)
            width = extra_args.get("width", 512)
            engine_name += f"_f{clip_frames}_{height}x{width}"
            
        engine_path = os.path.join(self.gpu_cache, f"{engine_name}.engine")
        if os.path.exists(engine_path) and not force_rebuild:
            print(f"[TRT] Found cached engine for {model_name}: {engine_path}")
            return engine_path

        print(f"[TRT] No cached engine for {model_name}. Building...")
        onnx_dir = self.gpu_cache
        
        if model_name == "denoising_unet_acc":
            from .export_unet import export_unet_to_onnx
            onnx_path = os.path.join(onnx_dir, f"{engine_name}.onnx")
            sim_onnx_path = os.path.join(onnx_dir, f"{engine_name}_sim.onnx")
            base_model_path = extra_args.get("base_model_path")
            clip_frames = extra_args.get("clip_frames", 12)
            video_length = extra_args.get("video_length", 25)
            height = extra_args.get("height", 512)
            width = extra_args.get("width", 512)
            
            # Export if the un-simplified onnx doesn't exist (we will simplify it externally or below)
            if not os.path.exists(sim_onnx_path):
                if not os.path.exists(onnx_path):
                    export_unet_to_onnx(pt_path, onnx_path, self.echomimic_dir, base_model_path=base_model_path, quantize_fp8=fp8, clip_frames=clip_frames, height=height, width=width)
                
                # Simplify the ONNX graph before building TRT engine
                print(f"[ONNX] Simplifying {onnx_path} to {sim_onnx_path}...")
                import subprocess
                import sys
                subprocess.run(
                    [sys.executable, "-m", "onnxsim", onnx_path, sim_onnx_path],
                    check=True
                )
            
            build_unet_engine(sim_onnx_path, engine_path, clip_frames=clip_frames, video_length=video_length, height=height, width=width, fp8=fp8)
            
        elif model_name.startswith("vae"):
            from .export_vae import export_vae_to_onnx
            export_vae_to_onnx(pt_path, onnx_dir) # Exports both encoder and decoder
            encoder_onnx = os.path.join(onnx_dir, "vae_encoder.onnx")
            decoder_onnx = os.path.join(onnx_dir, "vae_decoder.onnx")
            encoder_engine = os.path.join(onnx_dir, "vae_encoder.engine")
            decoder_engine = os.path.join(onnx_dir, "vae_decoder.engine")

            # Each builder must have its own network — sharing a network between two builds corrupts both engines.
            TRTEngineBuilder().build_engine(encoder_onnx, encoder_engine, dynamic_shapes={
                "input": [(1, 3, 512, 512), (1, 3, 512, 512), (2, 3, 512, 512)]
            })
            TRTEngineBuilder().build_engine(decoder_onnx, decoder_engine, dynamic_shapes={
                "latent": [(1, 4, 64, 64), (1, 4, 64, 64), (25, 4, 64, 64)]
            })
            return encoder_engine if "encoder" in model_name else decoder_engine

        elif model_name == "reference_unet":
            from .export_ref_unet import export_ref_unet_to_onnx
            onnx_path = os.path.join(onnx_dir, f"{model_name}.onnx")
            export_ref_unet_to_onnx(pt_path, onnx_path)
            builder = TRTEngineBuilder()
            builder.build_engine(onnx_path, engine_path, dynamic_shapes={
                "sample": [(1, 4, 64, 64), (1, 4, 64, 64), (1, 4, 64, 64)],
                "encoder_hidden_states": [(1, 1, 768), (1, 1, 768), (1, 1, 768)]
            })

        elif model_name == "pose_encoder":
            from .export_pose_encoder import export_pose_encoder_to_onnx
            onnx_path = os.path.join(onnx_dir, f"{model_name}.onnx")
            export_pose_encoder_to_onnx(pt_path, onnx_path, self.echomimic_dir)
            builder = TRTEngineBuilder()
            builder.build_engine(onnx_path, engine_path, dynamic_shapes={
                "conditioning": [(1, 3, 1, 512, 512), (1, 3, 12, 512, 512), (2, 3, 25, 512, 512)]
            })

        return engine_path

    def load_model(self, engine_path):
        return TRTModel(engine_path)
