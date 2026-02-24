"""Multi-GPU detection, pipeline management, and overlap-blend utilities.

When N >= 2 GPUs are available, pipelines are replicated across all GPUs and
video chunks are generated in parallel with K overlap frames blended via
linear crossfade.  When N == 1, the system falls back to sequential generation
with init_latent continuity (identical to the pre-multi-GPU behaviour).
"""

import os
import threading
import time
from typing import Optional

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# GPU Detection
# ---------------------------------------------------------------------------

def detect_gpus() -> int:
    """Return the number of available CUDA GPUs (0 if CUDA is not available)."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


# ---------------------------------------------------------------------------
# Overlap-Blend Utility
# ---------------------------------------------------------------------------

def blend_overlap(
    tail_video: np.ndarray,
    head_video: np.ndarray,
    overlap_frames: int,
) -> np.ndarray:
    """Linear crossfade between the last *overlap_frames* of *tail_video* and
    the first *overlap_frames* of *head_video*.

    Both inputs are expected in shape ``(1, C, F, H, W)`` with values in [0, 1].
    Returns a blended array of shape ``(1, C, overlap_frames, H, W)``.
    """
    if overlap_frames <= 0:
        return np.empty((tail_video.shape[0], tail_video.shape[1], 0,
                         tail_video.shape[3], tail_video.shape[4]),
                        dtype=tail_video.dtype)

    tail_slice = tail_video[:, :, -overlap_frames:, :, :]  # (1,C,K,H,W)
    head_slice = head_video[:, :, :overlap_frames, :, :]   # (1,C,K,H,W)

    # Build alpha ramp: shape (1, 1, K, 1, 1) so it broadcasts
    alpha = np.linspace(0.0, 1.0, overlap_frames, dtype=np.float32)
    alpha = alpha.reshape(1, 1, overlap_frames, 1, 1)

    blended = (1.0 - alpha) * tail_slice + alpha * head_slice
    return blended


# ---------------------------------------------------------------------------
# MultiGPUManager
# ---------------------------------------------------------------------------

class MultiGPUManager:
    """Manages N pipeline replicas, one per CUDA GPU.

    * ``num_gpus == 1`` → single-GPU mode (uses init_latent continuity)
    * ``num_gpus >= 2`` → multi-GPU mode (overlap-blend, no init_latent needed)
    """

    def __init__(
        self,
        config_path: str,
        echomimic_dir: str,
        weight_dtype: torch.dtype,
        audio_model_type: str = "whisper",
        overlap_frames: int = 6,
    ):
        self._num_gpus = max(1, detect_gpus())
        self.overlap_frames = overlap_frames if self._num_gpus >= 2 else 0
        self.weight_dtype = weight_dtype

        # Per-GPU state
        self.pipes = [None] * self._num_gpus
        self.devices = [f"cuda:{i}" for i in range(self._num_gpus)]
        self.reference_caches = [None] * self._num_gpus
        # For single-GPU mode: sequential latent continuity
        self.last_latents = [None] * self._num_gpus
        # For multi-GPU mode: tail frames buffer per GPU (last K decoded frames)
        self.tail_buffers: list[Optional[np.ndarray]] = [None] * self._num_gpus

        print(f"[GPU] Detected {self._num_gpus} CUDA GPU(s). "
              f"Overlap frames (K): {self.overlap_frames}")

        # Load pipelines (in parallel threads for speed)
        from core.models import load_pipeline

        if self._num_gpus == 1:
            print(f"[GPU] Loading pipeline on {self.devices[0]} ...")
            self.pipes[0] = load_pipeline(
                config_path, self.devices[0], weight_dtype, echomimic_dir,
                audio_model_type=audio_model_type,
            )
        else:
            threads = []
            errors = [None] * self._num_gpus

            def _load(idx):
                try:
                    print(f"[GPU] Loading pipeline on {self.devices[idx]} ...")
                    self.pipes[idx] = load_pipeline(
                        config_path, self.devices[idx], weight_dtype,
                        echomimic_dir, audio_model_type=audio_model_type,
                    )
                    print(f"[GPU] Pipeline {idx} ready on {self.devices[idx]}.")
                except Exception as e:
                    errors[idx] = e

            for i in range(self._num_gpus):
                t = threading.Thread(target=_load, args=(i,))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

            for i, err in enumerate(errors):
                if err is not None:
                    raise RuntimeError(
                        f"Failed to load pipeline on {self.devices[i]}: {err}"
                    )

        print(f"[GPU] All {self._num_gpus} pipeline(s) loaded.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_gpus(self) -> int:
        return self._num_gpus

    @property
    def is_multi_gpu(self) -> bool:
        return self._num_gpus >= 2

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_pipeline(self, gpu_idx: int):
        """Return ``(pipe, device_str)`` for the given GPU index."""
        idx = gpu_idx % self._num_gpus
        return self.pipes[idx], self.devices[idx]

    # ------------------------------------------------------------------
    # Reference Encoding
    # ------------------------------------------------------------------

    def encode_references(self, ref_image: Image.Image,
                          width: int, height: int,
                          steps: int, cfg: float):
        """Pre-compute and cache reference UNet states on every GPU."""
        for i in range(self._num_gpus):
            pipe = self.pipes[i]
            device = self.devices[i]
            print(f"[GPU] Encoding reference on {device} ...")
            self.reference_caches[i] = pipe.encode_reference(
                ref_image, width, height, steps, cfg,
                dtype=self.weight_dtype, device=device,
            )
        print("[GPU] Reference encoded on all GPUs.")

    # ------------------------------------------------------------------
    # Latent / Tail Buffer Management
    # ------------------------------------------------------------------

    def store_tail_buffer(self, gpu_idx: int, video_np: np.ndarray):
        """Store the last K frames of decoded video as tail buffer."""
        idx = gpu_idx % self._num_gpus
        if self.overlap_frames > 0 and video_np.shape[2] >= self.overlap_frames:
            self.tail_buffers[idx] = video_np[:, :, -self.overlap_frames:, :, :].copy()

    def get_and_clear_tail_buffer(self, gpu_idx: int) -> Optional[np.ndarray]:
        """Retrieve (and clear) the tail buffer for blending."""
        idx = gpu_idx % self._num_gpus
        buf = self.tail_buffers[idx]
        self.tail_buffers[idx] = None
        return buf

    def reset(self):
        """Reset all per-GPU state (call after silence)."""
        for i in range(self._num_gpus):
            self.last_latents[i] = None
            self.tail_buffers[i] = None
