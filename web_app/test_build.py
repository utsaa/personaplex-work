import sys
import os
sys.path.insert(0, "/workspace/personaplex-work/web_app")
sys.path.insert(0, "/workspace/personaplex-work/echomimic_v2")

from core.trt.builder import build_unet_engine
import tensorrt as trt

onnx_path = "/workspace/personaplex-work/echomimic_v2/engines/nvidia_geforce_rtx_4090/denoising_unet_acc_sim.onnx"
engine_path = "/workspace/personaplex-work/echomimic_v2/engines/nvidia_geforce_rtx_4090/denoising_unet_acc.engine"

print("Starting TRT Build Test...")
try:
    build_unet_engine(onnx_path, engine_path, clip_frames=12, video_length=13, fp8=False)
    print("Success!")
except Exception as e:
    print(f"Build Failed: {e}")
