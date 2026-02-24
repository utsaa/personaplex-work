
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
    # Draw into native-resolution canvas, then resize to target (W, H)
    native_h = max(re, im.shape[0] + rb)
    native_w = max(ce, im.shape[1] + cb)
    tgt_musk = np.zeros((native_h, native_w, 3), dtype=np.uint8)
    tgt_musk[rb:re, cb:ce, :] = im
    tgt_musk_pil = Image.fromarray(tgt_musk).convert("RGB").resize((W, H), Image.LANCZOS)
    
    # Return (C, H, W)
    return (
        torch.Tensor(np.array(tgt_musk_pil))
        .to(dtype=dtype, device=device)
        .permute(2, 0, 1) / 255.0
    )

class PoseProvider:
    def get_batch(self, start_idx: int, num_frames: int) -> torch.Tensor:
        """Returns (1, F, C, H, W) tensor batch."""
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
        
        # Stack (F, C, H, W) -> Permute (C, F, H, W) -> Unsqueeze (1, C, F, H, W)
        # Wait, original code was: stack(dim=1).unsqueeze(0).
        # Assuming prepare_pose_tensor returned (1, C, F, H, W).
        # Let's verify original code structure.
        # Original: torch.stack(pose_list, dim=1).unsqueeze(0)
        # pose_list elements are (C, H, W).
        # stack dim=1 implies (C, F, H, W). 
        # unsqueeze(0) implies (1, C, F, H, W). Correct.
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
             # Load but store on CPU
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
