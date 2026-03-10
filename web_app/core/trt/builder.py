import os
import tensorrt as trt
import torch

class TRTEngineBuilder:
    def __init__(self, logger_severity=trt.Logger.INFO):
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
            fp8 (bool): Whether to enable FP8 precision (requires TRT 9+ and H100/4090+ compatible hardware).
        """
        parser = trt.OnnxParser(self.network, self.logger)
        
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
            
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
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
            
        # Build and serialize the engine
        print(f"[TRT] Building engine: {engine_path} ... (This may take several minutes)")
        serialized_engine = self.builder.build_serialized_network(self.network, self.config)
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
            
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        print(f"[TRT] Engine built successfully: {engine_path}")
        return engine_path

def build_unet_engine(onnx_path, engine_path, batch_size=2, video_length=25, height=512, width=512, fp8=False):
    # Latent dimensions
    h_lat, w_lat = height // 8, width // 8
    
    # Dynamic shapes configuration
    # Batch can be 1 (single-shot) or 2 (cfg)
    # Frames can be 1 to video_length
    dynamic_shapes = {
        "sample": [(1, 4, 1, h_lat, w_lat), (batch_size, 4, video_length // 2, h_lat, w_lat), (batch_size, 4, video_length, h_lat, w_lat)],
        "timestep": [(1,), (batch_size,), (batch_size,)],
        # audio_cond_fea is 4D: (B, F, n_audio, cross_attention_dim=384)
        "audio_cond_fea": [(1, 1, 1, 384), (batch_size, video_length // 2, 1, 384), (batch_size, video_length, 1, 384)],
        # face_musk_fea is PoseEncoder output: (B, 320, F, H_lat, W_lat)
        "face_musk_fea": [(1, 320, 1, h_lat, w_lat), (batch_size, 320, video_length // 2, h_lat, w_lat), (batch_size, 320, video_length, h_lat, w_lat)]
    }
    
    builder = TRTEngineBuilder()
    return builder.build_engine(onnx_path, engine_path, dynamic_shapes=dynamic_shapes, fp8=fp8)
