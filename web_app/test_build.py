import sys
import os
import os
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_base_dir, "web_app"))
sys.path.insert(0, os.path.join(_base_dir, "echomimic_v2"))

from core.trt.builder import build_unet_engine
import tensorrt as trt

onnx_path = os.path.join(_base_dir, "echomimic_v2", "engines", "nvidia_geforce_rtx_4090", "denoising_unet_acc_sim.onnx")
engine_path = os.path.join(_base_dir, "echomimic_v2", "engines", "nvidia_geforce_rtx_4090", "denoising_unet_acc.engine")

print("Starting TRT Build Test...")
try:
    build_unet_engine(onnx_path, engine_path, batch_size=2, clip_frames=12, fp8=False)
    print("Success!")
except Exception as e:
    print(f"Build Failed: {e}")
