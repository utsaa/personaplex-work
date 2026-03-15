import os
import tensorrt as trt
import torch


class TRTEngineBuilder:
    def __init__(self, logger_severity=trt.Logger.VERBOSE):
        self.logger = trt.Logger(logger_severity)
        self.builder = trt.Builder(self.logger)
        self.config = self.builder.create_builder_config()
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    def build_engine(self, onnx_path, engine_path, dynamic_shapes=None, fp16=True, fp8=False):
        """
        Builds a TensorRT engine from an ONNX model.

        Args:
            onnx_path (str): Path to the ONNX model.
            engine_path (str): Path to save the built engine.
            dynamic_shapes (dict): Dictionary mapping input names to (min, opt, max) shapes.
            fp16 (bool): Whether to enable FP16 precision.
            fp8 (bool): Whether to enable FP8 precision (requires TRT 9+ and Ada Lovelace+).
        """
        parser = trt.OnnxParser(self.network, self.logger)

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # parse_from_file correctly handles external data files (.onnx + .onnx.data)
        # and avoids loading the entire 3GB model into Python memory.
        if not parser.parse_from_file(onnx_path):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

        if fp16:
            if not self.builder.platform_has_fast_fp16:
                print("[WARNING] FP16 not supported on this platform, falling back to FP32.")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)

        if fp8:
            if hasattr(trt, "BuilderFlag") and hasattr(trt.BuilderFlag, "FP8"):
                print("[TRT] Enabling FP8 precision flag.")
                self.config.set_flag(trt.BuilderFlag.FP8)
            else:
                print("[WARNING] FP8 not supported by this TensorRT version.")

        if dynamic_shapes:
            profile = self.builder.create_optimization_profile()
            for input_name, shapes in dynamic_shapes.items():
                min_shape, opt_shape, max_shape = shapes
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            self.config.add_optimization_profile(profile)

        # Limit workspace to 10GB to prevent "memSize >= 0" overflow
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 10 * 1024**3)
        if hasattr(trt, 'PreviewFeature') and hasattr(trt.PreviewFeature, 'DISABLE_EXTERNAL_TACTIC_CONTROL_FOR_CORE_TIMERS'):
            self.config.set_preview_feature(trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_CONTROL_FOR_CORE_TIMERS, True)

        # Allow fallback tactics and relax precision constraints (Fix for Error Code 9/10)
        self.config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)
        self.config.clear_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        
        # Advanced fusion breakers to stop "Giant Mystery Block" fusions
        if hasattr(trt.BuilderFlag, 'DISABLE_TACTIC_REGEN'):
            self.config.set_flag(trt.BuilderFlag.DISABLE_TACTIC_REGEN)

        # Restrict tactics to stable sources to bypass Myelin issues
        tactic_sources = (1 << int(trt.TacticSource.CUBLAS)) | \
                         (1 << int(trt.TacticSource.CUBLAS_LT)) | \
                         (1 << int(trt.TacticSource.CUDNN)) | \
                         (1 << int(trt.TacticSource.JIT_CONVOLUTIONS))
        if hasattr(trt.TacticSource, "EDGE_MASK_CONVOLUTIONS"):
             tactic_sources |= (1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS))
        self.config.set_tactic_sources(tactic_sources)

        # Build and serialize the engine
        print(f"[TRT] Building engine: {engine_path} ... (This may take several minutes)")
        serialized_engine = self.builder.build_serialized_network(self.network, self.config)

        if serialized_engine is None:
            # Check for specific network errors if build failed
            print("[ERROR] TensorRT engine build failed. Check network definition or precision constraints.")
            raise RuntimeError("Failed to build TensorRT engine")

        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        print(f"[TRT] Engine built successfully: {engine_path}")
        return engine_path


def build_unet_engine(onnx_path, engine_path, batch_size=2, clip_frames=12, height=512, width=512, fp8=False):
    """Builds a TRT engine for the UNet with correct dynamic shape profiles."""
    h_lat, w_lat = height // 8, width // 8
    # F = clip_frames + 1 reference frame.
    F_val = clip_frames + 1 

    # We set Min = Opt = Max to eliminate symbolic variable overhead in TRT autotuner.
    # This specifically addresses Error Code 10 by removing complex algebraic substutions.
    dynamic_shapes = {
        "sample":         [(batch_size, 4, F_val, h_lat, w_lat), (batch_size, 4, F_val, h_lat, w_lat), (batch_size, 4, F_val, h_lat, w_lat)],
        "timestep":       [(batch_size,), (batch_size,), (batch_size,)],
        "audio_cond_fea": [(batch_size, F_val, 1, 384), (batch_size, F_val, 1, 384), (batch_size, F_val, 1, 384)],
        "face_musk_fea":  [(batch_size, 320, F_val, h_lat, w_lat), (batch_size, 320, F_val, h_lat, w_lat), (batch_size, 320, F_val, h_lat, w_lat)],
    }

    builder = TRTEngineBuilder()
    return builder.build_engine(onnx_path, engine_path, dynamic_shapes=dynamic_shapes, fp8=fp8)


def build_pose_encoder_engine(onnx_path, engine_path, batch_size=1, clip_frames=12, height=512, width=512):
    """Builds a TRT engine for the Pose Encoder with semi-static shapes."""
    # Semi-static frames (fixed to clip_frames) to avoid InflatedConv3d conflicts.
    # Batch is dynamic (1-2) to match UNet requirements.
    dynamic_shapes = {
        "conditioning": [(batch_size, 3, clip_frames, height, width), (batch_size, 3, clip_frames, height, width), (batch_size, 3, clip_frames, height, width)]
    }

    builder = TRTEngineBuilder()
    return builder.build_engine(onnx_path, engine_path, dynamic_shapes=dynamic_shapes)
