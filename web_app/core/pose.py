import os
from typing import List

import torch
import numpy as np
from PIL import Image
from src.utils.dwpose_util import draw_pose_select_v2

def load_pose_files(pose_dir: str) -> List[str]:
    files = sorted(
        [f for f in os.listdir(pose_dir) if f.endswith(".npy")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    assert len(files) > 0, f"No .npy pose files found in {pose_dir}"
    return files

def get_pose_tensor(
    pose_dir: str, pose_file: str, 
    W: int, H: int, device: str, dtype: torch.dtype
) -> torch.Tensor:
    """Load a single pose tensor from disk (C, H, W)."""
    tgt_musk_path = os.path.join(pose_dir, pose_file)
    detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
    imh_new, imw_new, rb, re, cb, ce = detected_pose["draw_pose_params"]
    im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
    im = np.transpose(np.array(im), (1, 2, 0))
    
    # EXACT MATCH WITH INFER_ACC.PY!
    # Dynamically determine the required canvas size from the .npy boundaries
    # to support BOTH the 768x768 demo poses AND any custom 512x512 poses you make!
    canvas_w = max(W, ce)
    canvas_h = max(H, re)
    
    tgt_musk = np.zeros((canvas_h, canvas_w, 3)).astype('uint8')
    tgt_musk[rb:re, cb:ce, :] = im
    tgt_musk_pil = Image.fromarray(np.array(tgt_musk)).convert('RGB')
    
    # If the user requested a different target resolution, resize the fully drawn pose canvas
    if tgt_musk_pil.size != (W, H):
        tgt_musk_pil = tgt_musk_pil.resize((W, H))
    
    pose_tensor = torch.from_numpy(np.array(tgt_musk_pil).astype(np.float32) / 255.0).permute(2, 0, 1)
    return pose_tensor.to(dtype=dtype, device=device)

class PoseProvider:
    def get_batch(self, start_idx: int, num_frames: int) -> torch.Tensor:
        raise NotImplementedError

class OnTheFlyPoseProvider(PoseProvider):
    def __init__(
        self,
        pose_dir: str,
        pose_files: List[str],
        W: int,
        H: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self.pose_dir = pose_dir
        self.pose_files = pose_files
        self.W = W
        self.H = H
        self.device = device
        self.dtype = dtype

    def get_batch(self, start_idx: int, num_frames: int) -> torch.Tensor:
        pose_list = []
        num_available = len(self.pose_files)
        for i in range(num_frames):
            idx = (start_idx + i) % num_available
            pose_tensor = get_pose_tensor(
                self.pose_dir, self.pose_files[idx], 
                self.W, self.H, self.device, self.dtype
            )
            pose_list.append(pose_tensor)
        
        return torch.stack(pose_list, dim=1).unsqueeze(0)

class PreloadedPoseProvider(PoseProvider):
    def __init__(
        self,
        pose_dir: str,
        pose_files: List[str],
        W: int,
        H: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        print(f"[INIT] Pre-loading {len(pose_files)} pose tensors to CPU RAM...")
        self.device = device
        self.tensors: List[torch.Tensor] = []
        for i, f in enumerate(pose_files):
             t = get_pose_tensor(pose_dir, f, W, H, "cpu", dtype)
             self.tensors.append(t)
             if i % 50 == 0:
                 print(f"  Loaded {i}/{len(pose_files)}")
        print(f"[INIT] Finished pre-loading poses.")

    def get_batch(self, start_idx: int, num_frames: int) -> torch.Tensor:
        pose_list = []
        num_available = len(self.tensors)
        for i in range(num_frames):
            idx = (start_idx + i) % num_available
            pose_list.append(self.tensors[idx].to(self.device))
        
        return torch.stack(pose_list, dim=1).unsqueeze(0)
