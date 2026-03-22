import os
from dotenv import load_dotenv
import torch
import hashlib
import gc
from .builder import TRTEngineBuilder, build_unet_engine, build_pose_encoder_engine
from .runtime import TRTModel

load_dotenv()


class TRTEngineManager:
    def __init__(self, echomimic_dir, cache_dir="engines"):
        self.echomimic_dir = echomimic_dir
        self.cache_base = os.path.join(echomimic_dir, cache_dir)
        self.gpu_name = torch.cuda.get_device_name(0).replace(" ", "_").lower()
        self.gpu_cache = os.path.join(self.cache_base, self.gpu_name)
        os.makedirs(self.gpu_cache, exist_ok=True)

    def get_unet_engine(self, unet_pt_path, base_model_path=None, force_rebuild=False, fp8=False, clip_frames=12, video_length=25, width=512, height=512, use_safetensors=False):
        """Gets the path to a compiled UNet engine.
        
        Args:
            clip_frames: Number of audio clip frames (excl. reference). Matches fps * audio_margin.
            video_length: Maximum video length in frames.
        """
        return self._get_engine("denoising_unet_acc", unet_pt_path, force_rebuild,
                               extra_args={"base_model_path": base_model_path, "fp8": fp8,
                                           "clip_frames": clip_frames, "video_length": video_length,
                                           "width": width, "height": height, "use_safetensors": use_safetensors})

    def get_vae_encoder_engine(self, vae_path, force_rebuild=False, width=512, height=512, use_safetensors=False):
        return self._get_engine("vae_encoder", vae_path, force_rebuild, extra_args={"width": width, "height": height, "use_safetensors": use_safetensors})

    def get_vae_decoder_engine(self, vae_path, force_rebuild=False, width=512, height=512, use_safetensors=False):
        return self._get_engine("vae_decoder", vae_path, force_rebuild, extra_args={"width": width, "height": height, "use_safetensors": use_safetensors})

    def get_ref_unet_engine(self, ref_unet_pt_path, base_model_path=None, force_rebuild=False, width=512, height=512, use_safetensors=False):
        return self._get_engine("reference_unet", ref_unet_pt_path, force_rebuild,
                               extra_args={"base_model_path": base_model_path, "width": width, "height": height, "use_safetensors": use_safetensors})

    def get_pose_encoder_engine(self, pose_pt_path, force_rebuild=False, width=512, height=512, clip_frames=12):
        return self._get_engine("pose_encoder", pose_pt_path, force_rebuild, extra_args={"width": width, "height": height, "clip_frames": clip_frames})

    def _get_engine(self, model_name, pt_path, force_rebuild, extra_args=None):
        extra_args = extra_args or {}
        fp8 = extra_args.get("fp8", False)
        width = extra_args.get("width", 512)
        height = extra_args.get("height", 512)
        
        # Suffix engine path if FP8 or specific dimensions are used
        engine_name = model_name
        if fp8:
            engine_name += "_fp8"
            
        if model_name == "denoising_unet_acc":
            clip_frames = extra_args.get("clip_frames", 12)
            engine_name += f"_f{clip_frames}_{width}x{height}"
        elif model_name == "pose_encoder":
            clip_frames = extra_args.get("clip_frames", 12)
            engine_name += f"_f{clip_frames}_{width}x{height}"
        else:
            engine_name += f"_{width}x{height}"
            
        engine_path = os.path.join(self.gpu_cache, f"{engine_name}.engine")
        if os.path.exists(engine_path) and not force_rebuild:
            print(f"[TRT] Found cached engine for {model_name}: {engine_path}")
            return engine_path

        print(f"[TRT] No cached engine for {model_name}. Building...")
        onnx_dir = self.gpu_cache
        
        if model_name == "denoising_unet_acc":
            from .export_unet import export_unet_to_onnx
            onnx_path = os.path.join(onnx_dir, f"{engine_name}.onnx")
            base_model_path = extra_args.get("base_model_path")
            clip_frames = extra_args.get("clip_frames", 12)
            height = extra_args.get("height", 512)
            use_safetensors = extra_args.get("use_safetensors", False)
            
            # Export if the un-simplified onnx doesn't exist (we will simplify it externally or below)
            onnx_path = os.path.join(onnx_dir, f"denoising_unet_acc_f{clip_frames}_{width}x{height}.onnx")
            sim_onnx_path = os.path.join(onnx_dir, f"denoising_unet_acc_f{clip_frames}_{width}x{height}_sim.onnx")
            
            # Determine the best available ONNX file
            skip_simplify = os.environ.get("TRT_SKIP_SIMPLIFY", "false").lower() == "true"
            if skip_simplify:
                final_onnx = onnx_path
            else:
                final_onnx = sim_onnx_path if os.path.exists(sim_onnx_path) else onnx_path

            if not os.path.exists(final_onnx):
                # Neither simplified nor raw ONNX exists -> Full export flow
                print(f"[TRT] ONNX files missing. Starting export flow...")
                export_unet_to_onnx(pt_path, onnx_path, self.echomimic_dir, base_model_path=base_model_path, quantize_fp8=fp8, clip_frames=clip_frames, height=height, width=width, use_safetensors=use_safetensors)
                
                # Consolidate external data into a single .data file (cleaner than hundreds of .weight files)
                import onnx
                print(f"[ONNX] Consolidating external data for {onnx_path}...")
                model = onnx.load(onnx_path, load_external_data=True)
                onnx.save(model, onnx_path, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(onnx_path) + ".data")
                del model
                gc.collect()

                if skip_simplify:
                    print(f"[ONNX] Skipping simplification as TRT_SKIP_SIMPLIFY is set.")
                    final_onnx = onnx_path
                else:
                    # Try to simplify, but fallback if it fails (common for >2GB models due to protobuf limits)
                    print(f"[ONNX] Attempting to simplify {onnx_path} to {sim_onnx_path}...")
                    try:
                        import subprocess
                        import sys
                        subprocess.run(
                            [sys.executable, "-m", "onnxsim", onnx_path, sim_onnx_path, "--no-large-tensor"],
                            check=True
                        )
                        print(f"[ONNX] Simplification successful.")
                        final_onnx = sim_onnx_path
                    except Exception as e:
                        print(f"[ONNX] Simplification failed (likely due to model size): {e}")
                        print(f"[ONNX] Falling back to non-simplified ONNX.")
                        final_onnx = onnx_path

                gc.collect()
                torch.cuda.empty_cache()
            
            print(f"[TRT] Building engine from {final_onnx}...")
            build_unet_engine(final_onnx, engine_path, batch_size=2, clip_frames=clip_frames, height=height, width=width, fp8=fp8)
            
            # Free memory after large build
            gc.collect()
            torch.cuda.empty_cache()
            
        elif model_name.startswith("vae"):
            from .export_vae import export_vae_to_onnx
            use_safetensors = extra_args.get("use_safetensors", False)
            export_vae_to_onnx(pt_path, onnx_dir, height=height, width=width, use_safetensors=use_safetensors) # Exports both encoder and decoder
            gc.collect()
            torch.cuda.empty_cache()

            encoder_onnx = os.path.join(onnx_dir, f"vae_encoder_{width}x{height}.onnx")
            decoder_onnx = os.path.join(onnx_dir, f"vae_decoder_{width}x{height}.onnx")
            encoder_engine = os.path.join(self.gpu_cache, f"vae_encoder_{width}x{height}.engine")
            decoder_engine = os.path.join(self.gpu_cache, f"vae_decoder_{width}x{height}.engine")

            # Calculate latent shapes
            h_lat, w_lat = height // 8, width // 8

            # Each builder must have its own network — sharing a network between two builds corrupts both engines.
            TRTEngineBuilder().build_engine(encoder_onnx, encoder_engine, dynamic_shapes={
                "input": [(1, 3, height, width), (1, 3, height, width), (2, 3, height, width)]
            })
            TRTEngineBuilder().build_engine(decoder_onnx, decoder_engine, dynamic_shapes={
                "latent": [(1, 4, h_lat, w_lat), (1, 4, h_lat, w_lat), (32, 4, h_lat, w_lat)]
            })
            return encoder_engine if "encoder" in model_name else decoder_engine

        elif model_name == "reference_unet":
            from .export_ref_unet import export_ref_unet_to_onnx
            onnx_path = os.path.join(onnx_dir, f"{model_name}_{width}x{height}.onnx")
            base_model_path = extra_args.get("base_model_path")
            use_safetensors = extra_args.get("use_safetensors", False)
            export_ref_unet_to_onnx(pt_path, onnx_path, self.echomimic_dir, base_model_path=base_model_path, height=height, width=width, use_safetensors=use_safetensors)
            gc.collect()
            torch.cuda.empty_cache()
            
            h_lat, w_lat = height // 8, width // 8
            
            builder = TRTEngineBuilder()
            builder.build_engine(onnx_path, engine_path, dynamic_shapes={
                "sample": [(1, 4, h_lat, w_lat), (1, 4, h_lat, w_lat), (2, 4, h_lat, w_lat)],
                "timestep": [(1,), (1,), (2,)],
                "encoder_hidden_states": [(1, 1, 768), (1, 1, 768), (2, 1, 768)]
            })

        elif model_name == "pose_encoder":
            from .export_pose_encoder import export_pose_encoder_to_onnx
            clip_frames = extra_args.get("clip_frames", 12)
            onnx_path = os.path.join(onnx_dir, f"{model_name}_f{clip_frames}_{width}x{height}.onnx")
            if not os.path.exists(onnx_path):
                export_pose_encoder_to_onnx(pt_path, onnx_path, self.echomimic_dir, height=height, width=width, clip_frames=clip_frames)
                gc.collect()
                torch.cuda.empty_cache()
            
            # Use standardized builder with dynamic batch support
            build_pose_encoder_engine(onnx_path, engine_path, batch_size=2, clip_frames=clip_frames, height=height, width=width)

        return engine_path

    def load_model(self, engine_path):
        return TRTModel(engine_path)
