import sys
import os
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_base_dir, "web_app"))
sys.path.insert(0, os.path.join(_base_dir, "echomimic_v2"))

from core.trt.manager import TRTEngineManager

manager = TRTEngineManager(echomimic_dir=os.path.join(_base_dir, "echomimic_v2"))
unet_pt_path = os.path.join(_base_dir, "echomimic_v2", "pretrained_weights", "denoising_unet_acc.pth")
base_model_path = os.path.join(_base_dir, "echomimic_v2", "pretrained_weights", "sd-image-variations-diffusers")

print("Starting FULL TRT Export + Build Pipeline...")
for f in [12]:
    print(f"\n--- Building for clip_frames={f} ---")
    try:
        engine_path = manager.get_unet_engine(unet_pt_path, base_model_path=base_model_path, force_rebuild=True, clip_frames=f, fp8=False)
        print(f"Success! Engine for {f} frames at {engine_path}")
    except Exception as e:
        print(f"Build Failed for {f} frames: {e}")
        # Continue to next if one fails
        continue
