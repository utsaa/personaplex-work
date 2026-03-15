import os
import sys
import torch

# Add current directory to path
sys.path.append(os.getcwd())

from core.trt.manager import TRTEngineManager

echomimic_dir = "/workspace/personaplex-work/echomimic_v2"
pose_pt_path = os.path.join(echomimic_dir, "pretrained_weights/pose_encoder.pth")

manager = TRTEngineManager(echomimic_dir)
print("Starting Pose Encoder engine build...")
try:
    # Force rebuild to see the process
    engine_path = manager.get_pose_encoder_engine(pose_pt_path, force_rebuild=True, width=512, height=512, clip_frames=12)
    print(f"Success! Engine built at: {engine_path}")
except Exception as e:
    print(f"Failed with error: {e}")
    import traceback
    traceback.print_exc()
