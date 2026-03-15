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
# Run-log (System FD Redirection)
# ---------------------------------------------------------------------------

def setup_logging() -> str:
    """
    Redirects system file descriptors 1 (stdout) and 2 (stderr) to a log file
    while also 'tee-ing' them back to the terminal. This ensures that
    low-level C++ and CUDA library errors (like TensorRT) are captured in 
    the log file, which Python-level redirection (sys.stdout) often misses.
    """
    import os
    import threading
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"run_{timestamp}.log")
    
    # Save original terminal FDs (so we can still write to them)
    try:
        orig_stdout_fd = os.dup(1)
        orig_stderr_fd = os.dup(2)
    except Exception as e:
        print(f"[WARN] Failed to duplicate FDs: {e}. Logging may not be complete.")
        return ""

    log_file = open(log_path, "w", encoding="utf-8")

    def tee_fd(target_fd, original_fd, log_f):
        r, w = os.pipe()
        os.dup2(w, target_fd)
        os.close(w)

        def logger_thread():
            while True:
                try:
                    data = os.read(r, 4096)
                    if not data:
                        break
                    # Write to terminal
                    os.write(original_fd, data)
                    # Write to log file
                    text = data.decode(errors='replace')
                    log_f.write(text)
                    log_f.flush()
                except Exception:
                    break
        
        t = threading.Thread(target=logger_thread, daemon=True)
        t.start()

    # Redirect stdout and stderr
    tee_fd(1, orig_stdout_fd, log_file)
    tee_fd(2, orig_stderr_fd, log_file)

    print(f"[LOG] Logging system FDs to {log_path}")
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
    parser.add_argument("--use-trt", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable TensorRT accelerated inference (disabled by default). Use --use-trt to enable.")
    parser.add_argument("--use-blend", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable overlap-blending on single GPU (disables init-latent).")
    
    args = parser.parse_args()

    setup_logging()
    print(f"[INIT] Arguments: {args}")

    if getattr(args, "use_blend", False):
        print("[INIT] --use-blend active. Disabling --use-init-latent for temporal stability.")
        args.use_init_latent = False

    # GPU Detection & Pipeline Loading
    n_gpus = detect_gpus()
    print(f"[INIT] Detected {n_gpus} CUDA GPU(s).")

    gpu_manager = MultiGPUManager(
        config_path=args.config,
        echomimic_dir=_ECHOMIMIC_DIR,
        weight_dtype=WEIGHT_DTYPE,
        audio_model_type=args.audio_model_type,
        overlap_frames=args.overlap_frames,
        use_trt=args.use_trt,
        fp8=args.quantize_fp8,
        force_blend=args.use_blend,
        clip_frames=args.clip_frames,
        width=args.width,
        height=args.height,
    )

    # Quantization (apply to all GPUs for non-TRT path)
    if args.quantize_fp8 and not args.use_trt:
        for i in range(gpu_manager.num_gpus):
            pipe, device = gpu_manager.get_pipeline(i)
            try:
                from torchao.quantization import quantize_, Float8WeightOnlyConfig
                print(f"[INIT] Quantizing Denoising UNet to FP8 (PyTorch) on {device}...")
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
        # Increase recompile limit and disable static parameter shapes 
        # to prevent eager fallback from ResNet channel variations.
        torch._dynamo.config.force_parameter_static_shapes = False
        torch._dynamo.config.recompile_limit = 120

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
