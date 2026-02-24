"""EchoMimic-v2 web app (Multi-GPU).

Serves a browser UI that streams mic audio via WebSocket, runs it through
the accelerated EchoMimic-v2 diffusion pipeline on the server, and streams
back tagged JPEG video frames + PCM audio for synchronised playback.

Supports 1..N CUDA GPUs automatically:
  - N == 1: sequential generation with init_latent continuity
  - N >= 2: pipelined overlap-blend for parallel GPU usage
"""

import argparse
import asyncio
import os
import sys
import torch
try:
    import torch._dynamo
except ImportError:
    pass
import warnings
from datetime import datetime
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure echomimic_v2 is importable (sibling directory)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ECHOMIMIC_DIR = os.path.join(os.path.dirname(_HERE), "echomimic_v2")
if _ECHOMIMIC_DIR not in sys.path:
    sys.path.insert(0, _ECHOMIMIC_DIR)

# Import local modules (after sys.path fix)
from core.gpu import MultiGPUManager, detect_gpus
from core.pose import load_pose_files, PreloadedPoseProvider, OnTheFlyPoseProvider
from core.server import run_server
from core.monitoring import PerformanceMonitor

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(_ECHOMIMIC_DIR, "configs", "prompts", "infer_acc.yaml")
DEFAULT_REF_IMAGE = os.path.join(_ECHOMIMIC_DIR, "assets", "therapist_ref.png")
DEFAULT_POSE_DIR = os.path.join(_ECHOMIMIC_DIR, "assets", "halfbody_demo", "pose", "01")
INDEX_HTML_PATH = os.path.join(_HERE, "index.html")
LOGS_DIR = os.path.join(_HERE, "logs")

WEIGHT_DTYPE = torch.float16


# ---------------------------------------------------------------------------
# Run-log
# ---------------------------------------------------------------------------
class TeeStream:
    """Write to both the original stream and a log file simultaneously."""
    def __init__(self, original_stream: object, log_file: object) -> None:
        self._original = original_stream
        self._log = log_file

    def write(self, data: str) -> None:
        self._original.write(data)
        self._original.flush()
        try:
            self._log.write(data)
            self._log.flush()
        except Exception:
            pass

    def flush(self) -> None:
        self._original.flush()
        try:
            self._log.flush()
        except Exception:
            pass

    def __getattr__(self, name: str) -> object:
        return getattr(self._original, name)


def setup_logging() -> str:
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"run_{timestamp}.log")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)
    print(f"[LOG] Logging to {log_path}")
    return log_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EchoMimic-v2 web app (browser audio -> pipeline -> browser video+audio)",
    )
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument("--reference-image", type=str, default=DEFAULT_REF_IMAGE)
    parser.add_argument("--pose-dir", type=str, default=DEFAULT_POSE_DIR)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--clip-frames", type=int, default=12)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--cfg", type=float, default=2.5)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--vad-threshold", type=float, default=0.015,
                        help="Server-side RMS silence threshold (0.0 = disabled). "
                             "Clips below this are discarded.")
    parser.add_argument("--use-init-latent", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable latent state preservation for continuity between clips. "
                             "Use --no-use-init-latent to disable (old behavior).")
    parser.add_argument("--audio-margin", type=int, default=2,
                        help="Audio feature context margin (frames). Higher = more context for lip sync.")
    parser.add_argument("--compile-unet", action="store_true",
                        help="Enable torch.compile for the denoising UNet.")
    parser.add_argument("--compile-unet-mode", type=str, default="reduce-overhead",
                        help="Mode for torch.compile. Valid values: "
                             "'default' (standard), "
                             "'reduce-overhead' (faster startup, stable performance - RECOMMENDED), "
                             "'max-autotune' (intensive benchmarking, fastest inference), "
                             "'max-autotune-no-cudagraphs' (benchmarking without CUDA graphs).")
    parser.add_argument("--quantize-fp8", action=argparse.BooleanOptionalAction, default=False,
                        help="Quantize UNet to FP8 (requires torchao & L4/H100/4090 GPU).")
    parser.add_argument("--low-ram", action="store_true",
                        help="Disable pose pre-loading (saves ~1GB RAM but increases CPU latency).")
    parser.add_argument("--audio-model-type", type=str, default="whisper", choices=["whisper", "wav2vec2"],
                        help="Type of audio model to use for feature extraction.")
    parser.add_argument("--overlap-frames", type=int, default=6,
                        help="Number of overlap frames (K) for multi-GPU blending. "
                             "Only used when >=2 GPUs are detected. Default: 6.")
    
    args = parser.parse_args()

    setup_logging()

    # GPU Detection & Pipeline Loading
    n_gpus = detect_gpus()
    print(f"[INIT] Detected {n_gpus} CUDA GPU(s).")

    gpu_manager = MultiGPUManager(
        config_path=args.config,
        echomimic_dir=_ECHOMIMIC_DIR,
        weight_dtype=WEIGHT_DTYPE,
        audio_model_type=args.audio_model_type,
        overlap_frames=args.overlap_frames,
    )

    # Quantization (apply to all GPUs)
    if args.quantize_fp8:
        for i in range(gpu_manager.num_gpus):
            pipe, device = gpu_manager.get_pipeline(i)
            try:
                from torchao.quantization import quantize_, Float8WeightOnlyConfig
                print(f"[INIT] Quantizing Denoising UNet to FP8 on {device}...")
                quantize_(pipe.denoising_unet, Float8WeightOnlyConfig())
            except ImportError:
                try:
                    from torchao.quantization import quantize_, float8_weight_only
                    print(f"[INIT] Using float8_weight_only alias on {device}...")
                    quantize_(pipe.denoising_unet, float8_weight_only())
                except ImportError:
                    print("[ERROR] torchao not found. Please install via 'uv pip install torchao'")
            except Exception as e:
                print(f"[ERROR] FP8 Quantization failed on {device}: {e}")

    # Compilation (apply to all GPUs)
    if args.compile_unet:
        valid_modes = ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]
        if args.compile_unet_mode not in valid_modes:
            raise ValueError(f"Invalid mode --compile-unet-mode='{args.compile_unet_mode}'. "
                             f"Must be one of: {valid_modes}")

        for i in range(gpu_manager.num_gpus):
            pipe, device = gpu_manager.get_pipeline(i)
            print(f"[INIT] Compiling UNet on {device} (mode={args.compile_unet_mode})...")
            
            backend = "inductor"
            if os.name == 'nt':
                print("[WARN] Windows detected. Using default backend selection.")
                backend = None
                if hasattr(torch, "_dynamo"):
                    torch._dynamo.config.suppress_errors = True
                
            try:
                compile_kwargs = {
                    "mode": args.compile_unet_mode,
                    "fullgraph": False
                }
                if backend:
                    compile_kwargs["backend"] = backend

                pipe.denoising_unet = torch.compile(pipe.denoising_unet, **compile_kwargs)
            except Exception as e:
                print(f"[WARN] Compilation failed on {device}: {e}. Running in eager mode.")

    # Validation
    if not os.path.exists(args.reference_image):
        print(f"[ERROR] Not found: {args.reference_image}")
        sys.exit(1)
    ref_image = Image.open(args.reference_image).convert("RGB")
    print(f"[INIT] Reference image: {args.reference_image}")

    if not os.path.isdir(args.pose_dir):
        print(f"[ERROR] Not found: {args.pose_dir}")
        sys.exit(1)
    pose_files = load_pose_files(args.pose_dir)
    print(f"[INIT] Loaded {len(pose_files)} pose files from {args.pose_dir}")

    # Determine device for pose loading (use first GPU's device)
    pose_device = gpu_manager.devices[0]
    posedata_args = (args.pose_dir, pose_files, args.width, args.height, pose_device, WEIGHT_DTYPE)
    if args.low_ram:
        print("[INIT] Low RAM mode: using On-The-Fly pose loading.")
        pose_provider = OnTheFlyPoseProvider(*posedata_args)
    else:
        try:
            pose_provider = PreloadedPoseProvider(*posedata_args)
        except MemoryError:
            print("[WARN] MemoryError during pose pre-loading! Falling back to On-The-Fly loading.")
            pose_provider = OnTheFlyPoseProvider(*posedata_args)

    # Performance Monitoring
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    perf_log_dir = os.path.join(_HERE, "core", "performance_logs")
    perf_monitor = PerformanceMonitor(perf_log_dir, run_timestamp)
    perf_monitor.start(interval=1.0)

    # Run Server
    try:
        with torch.no_grad():
            asyncio.run(run_server(gpu_manager, ref_image, pose_provider, args, INDEX_HTML_PATH))
    finally:
        perf_monitor.stop()


if __name__ == "__main__":
    main()
