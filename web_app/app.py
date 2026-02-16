"""EchoMimic-v2 web app.

Serves a browser UI that streams mic audio via WebSocket, runs it through
the accelerated EchoMimic-v2 diffusion pipeline on the server, and streams
back tagged JPEG video frames + PCM audio for synchronised playback.

Usage (on RunPod or any headless GPU box):
    cd <parent_dir>              # e.g. the dir containing echomimic_v2/
    python -m web_app.app --port 8080

Architecture:
  Browser ──(WS: float32 PCM)──► Server ──► video_generation_thread
  Browser ◄──(WS: 0x01+JPEG / 0x02+PCM)── Server ◄── frame_queue / audio_out_queue
"""

import argparse
import asyncio
from datetime import datetime
import os
import queue
import random
import sys
import tempfile
import threading
import time
import wave

import cv2
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Ensure echomimic_v2 is importable (sibling directory)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ECHOMIMIC_DIR = os.path.join(os.path.dirname(_HERE), "echomimic_v2")
if _ECHOMIMIC_DIR not in sys.path:
    sys.path.insert(0, _ECHOMIMIC_DIR)

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import is_accelerate_available
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2_acc import EchoMimicV2Pipeline
from src.models.pose_encoder import PoseEncoder
from src.utils.dwpose_util import draw_pose_select_v2

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(_ECHOMIMIC_DIR, "configs", "prompts", "infer_acc.yaml")
DEFAULT_REF_IMAGE = os.path.join(_ECHOMIMIC_DIR, "assets", "therapist_ref.png")
DEFAULT_POSE_DIR = os.path.join(_ECHOMIMIC_DIR, "assets", "halfbody_demo", "pose", "01")
INDEX_HTML_PATH = os.path.join(_HERE, "index.html")
LOGS_DIR = os.path.join(_HERE, "logs")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DTYPE = torch.float16


# ---------------------------------------------------------------------------
# Run-log: tee stdout/stderr to a file inside web_app/logs/
# ---------------------------------------------------------------------------

class TeeStream:
    """Write to both the original stream and a log file simultaneously."""

    def __init__(self, original_stream, log_file):
        self._original = original_stream
        self._log = log_file

    def write(self, data):
        self._original.write(data)
        self._original.flush()
        try:
            self._log.write(data)
            self._log.flush()
        except Exception:
            pass

    def flush(self):
        self._original.flush()
        try:
            self._log.flush()
        except Exception:
            pass

    # Forward everything else (fileno, isatty, etc.) to the original stream
    def __getattr__(self, name):
        return getattr(self._original, name)


def setup_logging() -> str:
    """Create logs/ dir and redirect stdout+stderr to a timestamped run log.

    Returns the path to the log file.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"run_{timestamp}.log")
    log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)
    print(f"[LOG] Logging to {log_path}")
    return log_path


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pipeline(config_path: str, device: str, weight_dtype: torch.dtype) -> EchoMimicV2Pipeline:
    """Load all EchoMimic-v2 ACC models and return an assembled pipeline."""
    print("[INIT] Loading EchoMimic-v2 (ACC) models ...")
    config = OmegaConf.load(config_path)
    infer_config = OmegaConf.load(
        os.path.join(_ECHOMIMIC_DIR, config.inference_config)
        if not os.path.isabs(config.inference_config)
        else config.inference_config
    )

    def _resolve(path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(_ECHOMIMIC_DIR, path)

    # VAE
    print("  loading VAE ...")
    vae = AutoencoderKL.from_pretrained(
        _resolve(config.pretrained_vae_path), local_files_only=True, torch_dtype=weight_dtype,
    ).to(device=device, dtype=weight_dtype)

    # Reference UNet (2D)
    print("  loading reference UNet ...")
    reference_unet = UNet2DConditionModel.from_pretrained(
        _resolve(config.pretrained_base_model_path), subfolder="unet",
    ).to(device=device, dtype=weight_dtype)
    reference_unet.load_state_dict(
        torch.load(_resolve(config.reference_unet_path), map_location="cpu"),
    )

    # Denoising UNet (3D + motion module)
    print("  loading denoising UNet (ACC) ...")
    motion_path = _resolve(config.motion_module_path)
    base_path = _resolve(config.pretrained_base_model_path)
    if os.path.exists(motion_path):
        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            base_path, motion_path, subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs),
        ).to(dtype=weight_dtype, device=device)
    else:
        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            base_path, "", subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
                "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim,
            },
        ).to(dtype=weight_dtype, device=device)
    denoising_unet.load_state_dict(
        torch.load(_resolve(config.denoising_unet_path), map_location="cpu"), strict=False,
    )

    # Pose encoder
    print("  loading pose encoder ...")
    pose_net = PoseEncoder(
        320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256),
    ).to(device=device, dtype=weight_dtype)
    pose_net.load_state_dict(
        torch.load(_resolve(config.pose_encoder_path), map_location="cpu"),
    )

    # Audio processor (Whisper tiny)
    print("  loading audio processor (Whisper tiny) ...")
    audio_processor = load_audio_model(
        model_path=_resolve(config.audio_model_path), device=device,
    )

    # Scheduler
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    # Assemble
    pipe = EchoMimicV2Pipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        pose_encoder=pose_net,
        scheduler=scheduler,
    )
    pipe = pipe.to(device, dtype=weight_dtype)

    # NEW: Enable Flash Attention / Xformers if available
    if is_accelerate_available():
        print("[INIT] Enabling xformers memory efficient attention (Flash Attention)...")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"[WARN] Failed to enable xformers: {e}")
    
    print("[READY] Pipeline loaded.\n")
    return pipe


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def load_pose_files(pose_dir: str) -> list[str]:
    files = sorted(
        [f for f in os.listdir(pose_dir) if f.endswith(".npy")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    assert len(files) > 0, f"No .npy pose files found in {pose_dir}"
    return files


def prepare_pose_tensor(
    pose_dir: str, pose_files: list[str],
    num_frames: int, start_idx: int,
    W: int, H: int, device: str, dtype: torch.dtype,
) -> torch.Tensor:
    num_available = len(pose_files)
    pose_list = []
    for i in range(num_frames):
        idx = (start_idx + i) % num_available
        tgt_musk_path = os.path.join(pose_dir, pose_files[idx])
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
        pose_list.append(
            torch.Tensor(np.array(tgt_musk_pil))
            .to(dtype=dtype, device=device)
            .permute(2, 0, 1) / 255.0
        )
    return torch.stack(pose_list, dim=1).unsqueeze(0)


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def generate_video_clip(
    pipe, ref_image, wav_path, poses_tensor,
    W, H, clip_frames, sample_rate, fps,
    steps=4, cfg=1.0, init_latent=None, use_init_latent=True,
    audio_margin=2,  # NEW: Audio context margin
    reference_cache=None,  # NEW: Cached reference states
    audio_context_frames=0,  # NEW: Rolling audio context
) -> tuple[np.ndarray | None, torch.Tensor | None]:
    """Generate a video clip and return both video frames and final latent.
    
    This function wraps the EchoMimic-v2 pipeline to generate a single clip.
    When use_init_latent=True and init_latent is provided, the first frame
    is initialized from the previous clip's final latent, ensuring continuity.
    
    Args:
        pipe: EchoMimic-v2 pipeline instance
        ref_image: Reference face image (PIL Image)
        wav_path: Path to temporary audio file for this clip
        poses_tensor: Pose guidance tensor for all frames
        W, H: Output dimensions in pixels
        clip_frames: Number of frames to generate
        sample_rate: Audio sample rate (Hz)
        fps: Frames per second
        steps: Number of diffusion denoising steps
        cfg: Classifier-free guidance scale
        init_latent: Optional latent tensor from previous clip's final frame.
                    Shape: (batch, channels, height, width) = (1, 4, 64, 64).
                    When provided and use_init_latent=True, this initializes
                    the first frame to ensure smooth continuity between clips.
        use_init_latent: Boolean toggle for latent state preservation.
                        If True, use init_latent for first frame initialization.
                        If False, generate all frames from random noise (old behavior).
        
    Returns:
        tuple: (video_np, final_latent)
            - video_np: Generated video frames as numpy array, shape (1, 3, frames, H, W)
            - final_latent: Latent tensor of the last frame for use in next clip
    """
    generator = torch.manual_seed(random.randint(0, 2**32 - 1))
    # Pipeline run with caching and audio context
    result = pipe(
        ref_image, wav_path,
        poses_tensor[:, :, :clip_frames, ...],
        W, H, clip_frames, steps, cfg,
        generator=generator,
        audio_sample_rate=sample_rate,
        context_frames=12, fps=fps,
        context_overlap=3, start_idx=0,
        audio_margin=audio_margin, 
        init_latents=init_latent if use_init_latent else None,
        reference_cache=reference_cache,  # NEW: Pass cache
        audio_context_frames=audio_context_frames,  # NEW: Pass context frames to trim
    )
    video = result.videos
    final_latent = result.final_latent  # NEW: Extract for next clip
    
    if isinstance(video, torch.Tensor):
        video_np = video.cpu().numpy()
    else:
        video_np = video
    
    return video_np, final_latent


# ---------------------------------------------------------------------------
# Thread: video generation (audio_queue -> diffusion -> frame_queue)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Utility: RMS energy of an audio buffer
# ---------------------------------------------------------------------------

def _rms(audio: np.ndarray) -> float:
    """Return the root-mean-square energy of a float32 PCM buffer."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def video_generation_thread(
    pipe, ref_image, pose_dir, pose_files,
    audio_queue, raw_clip_queue, stop_event,
    reference_cache,  # NEW: Receive cached reference states
    sample_rate=16000, fps=24, clip_frames=12,
    W=512, H=512, steps=4, cfg=1.0,
    vad_threshold=0.005, use_init_latent=True, audio_margin=2,
):
    """Diffusion-only thread: audio → pipeline → raw_clip_queue.
    """
    samples_per_clip = int(sample_rate * clip_frames / fps)
    
    # NEW: Audio Context Buffer
    # We want 1.0s of history + 0.5s of new audio = 1.5s total window
    history_samples = sample_rate * 1  # 1.0s history
    # The buffer will hold [history] + [current_clip_accumulation]
    audio_history_buffer = np.zeros(history_samples, dtype=np.float32)
    
    # Accumulation buffer for the *next* clip
    incoming_audio_buffer = np.array([], dtype=np.float32)

    pose_idx = 0
    last_latent = None 
    
    consecutive_silent_clips = 0
    idle_clips_limit = max(1, int(1.0 * fps / clip_frames))  # how many clips in 1s

    print(f"[GEN] Waiting for audio (target clip: {samples_per_clip} samples) ...")
    print(f"[GEN] Audio Context: Rolling 1.5s window (1.0s history + 0.5s new)")
    print(f"[GEN] Silence timeout: 1.0s (stop updating after {idle_clips_limit} silent clips)")

    while not stop_event.is_set():
        # Drain audio_queue into incoming buffer
        try:
            while True:
                chunk = audio_queue.get(timeout=0.05)
                incoming_audio_buffer = np.concatenate((incoming_audio_buffer, chunk))
        except queue.Empty:
            pass

        # 1. Backlog Catch-up: if we are > 1.5s behind, skip old audio to stay live
        if len(incoming_audio_buffer) > (samples_per_clip * 3):
            dropped_count = len(incoming_audio_buffer) - samples_per_clip
            print(f"[GEN] Backlog detected! Dropping {dropped_count} samples ({dropped_count/sample_rate:.2f}s) to keep up.")
            incoming_audio_buffer = incoming_audio_buffer[-samples_per_clip:]

        # Check if we have enough "new" audio for a clip
        if len(incoming_audio_buffer) < samples_per_clip:
            continue

        # Extract the exact 0.5s clip
        current_clip_audio = incoming_audio_buffer[:samples_per_clip]
        incoming_audio_buffer = incoming_audio_buffer[samples_per_clip:]

        # --- Server-side silence gate (on new audio only) ------------------
        clip_rms = _rms(current_clip_audio)
        
        if vad_threshold > 0 and clip_rms < vad_threshold:
            consecutive_silent_clips += 1
            # Only update history/log for the first second of silence
            if consecutive_silent_clips <= idle_clips_limit:
                print(f"[GEN] Silent clip (RMS={clip_rms:.5f} < {vad_threshold:.5f}) - Discarded.")
                audio_history_buffer = np.concatenate((audio_history_buffer, current_clip_audio))[-history_samples:]
            else:
                # Periodic idle message
                if consecutive_silent_clips % (idle_clips_limit * 6) == 0: # roughly every 6s
                     print(f"[GEN] System idle (RMS={clip_rms:.5f})...")
                
                # If silent for a long time, reset latent continuity to avoid artifacts on next speech
                if consecutive_silent_clips > (idle_clips_limit * 4): # > 4s
                    last_latent = None

            continue
        
        # We got speech!
        if consecutive_silent_clips > 0:
             print(f"[GEN] Audio detected (RMS={clip_rms:.5f} > {vad_threshold:.5f}). Generation starting...")
        consecutive_silent_clips = 0
        # -------------------------------------------------------------------

        # Prepare the full 1.5s window for Whisper
        # window = [history (1.0s)] + [current (0.5s)]
        full_audio_window = np.concatenate((audio_history_buffer, current_clip_audio))
        
        # Calculate how many audio frames correspond to the history (to tell pipeline to skip them)
        # Whisper (tiny) typically outputs 50 frames per second. 
        # But we pass `audio_context_frames` to the pipeline, which iterates over `whisper_chunks`.
        # `whisper_chunks` size depends on fps.
        # Logic: output video frames = clip_frames. 
        # The pipeline aligns audio chunks to video frames. 
        # If we pass 1.5s of audio, we get ~36 video frames worth of audio features.
        # We want to skip the first ~24 frames (1.0s) and generate the last 12 frames (0.5s).
        # Precise calculation: 
        # context_duration = len(audio_history_buffer) / sample_rate
        # context_video_frames = int(context_duration * fps)
        
        context_duration = len(audio_history_buffer) / sample_rate
        context_video_frames = int(context_duration * fps)

        tmp_wav_path = None
        try:
            tmp_fd, tmp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(tmp_fd)
            with wave.open(tmp_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                pcm_int16 = np.clip(full_audio_window * 32767, -32768, 32767).astype(np.int16)
                wf.writeframes(pcm_int16.tobytes())

            poses_tensor = prepare_pose_tensor(
                pose_dir, pose_files, clip_frames, pose_idx, W, H,
                device=DEVICE, dtype=WEIGHT_DTYPE,
            )
            pose_idx = (pose_idx + clip_frames) % len(pose_files)

            t0 = time.perf_counter()
            # Generate
            video_np, final_latent = generate_video_clip(
                pipe, ref_image, tmp_wav_path, poses_tensor,
                W, H, clip_frames, sample_rate, fps, steps, cfg,
                init_latent=last_latent,
                use_init_latent=use_init_latent,
                audio_margin=audio_margin,
                reference_cache=reference_cache, # NEW
                audio_context_frames=context_video_frames, # NEW
            )
            if use_init_latent:
                last_latent = final_latent 
            
            dt = time.perf_counter() - t0
            print(f"[GEN] Clip generated in {dt:.2f}s "
                  f"({clip_frames} frames). Context: {context_video_frames} frames skipped.")

            # Append current clip to history (rolling buffer)
            audio_history_buffer = np.concatenate((audio_history_buffer, current_clip_audio))[-history_samples:]

            # Hand off raw output
            if video_np is not None:
                raw_clip_queue.put((video_np, current_clip_audio.copy())) # Send ONLY new audio

        except Exception as e:
            print(f"[GEN] Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            if tmp_wav_path is not None:
                try:
                    os.unlink(tmp_wav_path)
                except OSError:
                    pass


def postprocess_thread(
    raw_clip_queue, socket_queue, stop_event,
):
    """Convert raw video tensors to BGR frames and push to delivery queues.

    Runs concurrently with the generation thread so that clip N+1's
    diffusion overlaps with clip N's post-processing and WebSocket send.
    """
    print("[POST] Post-processing thread started.")
    while not stop_event.is_set():
        try:
            video_np, clip_audio = raw_clip_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        t0 = time.perf_counter()

        # 1. Push audio for this clip (Priority 1)
        # We send audio FIRST so the client can schedule it immediately.
        if socket_queue is not None:
            try:
                # Prefix 0x02 = Audio
                msg = b'\x02' + clip_audio.tobytes()
                socket_queue.put(msg, timeout=0.1)
            except queue.Full:
                print("[POST] Socket queue full (dropping audio!)")
                pass

        # 2. Convert and push each frame (Priority 2)
        n_frames = video_np.shape[2]
        for f_idx in range(n_frames):
            if stop_event.is_set():
                return
            frame = video_np[0, :, f_idx, :, :]
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            frame = frame.transpose(1, 2, 0)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Encode JPEG
            _, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            try:
                # Prefix 0x01 = Video
                msg = b'\x01' + jpeg.tobytes()
                socket_queue.put(msg, timeout=0.1)
            except queue.Full:
                # Dropping frames is better than blocking generation
                print("[POST] Socket queue full (dropping frame)")
                pass

        dt = time.perf_counter() - t0
        print(f"[POST] {n_frames} frames post-processed & scheduled in {dt*1000:.1f}ms")


# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------

async def run_server(pipe, ref_image, pose_dir, pose_files, args):
    from aiohttp import web

    # NEW: Encode reference once at startup
    print(f"[INIT] Caching reference UNet states...")
    reference_cache = pipe.encode_reference(
        ref_image, args.width, args.height, args.steps, args.cfg,
        dtype=pipe.dtype, device=pipe.device
    )
    print(f"[INIT] Reference cached.")

    audio_queue = queue.Queue()
    raw_clip_queue = queue.Queue(maxsize=4)
    socket_queue = queue.Queue(maxsize=300)
    stop_event = threading.Event()

    gen_thread = threading.Thread(
        target=video_generation_thread,
        args=(
            pipe, ref_image, pose_dir, pose_files,
            audio_queue, raw_clip_queue, stop_event,
            reference_cache,
            args.sample_rate, args.fps, args.clip_frames,
            args.width, args.height, args.steps, args.cfg,
        ),
        kwargs={
            "vad_threshold": args.vad_threshold,
            "use_init_latent": args.use_init_latent,
            "audio_margin": args.audio_margin,  
        },
        daemon=True,
    )
    gen_thread.start()

    post_thread = threading.Thread(
        target=postprocess_thread,
        args=(raw_clip_queue, socket_queue, stop_event),
        daemon=True,
    )
    post_thread.start()

    # Read index.html once at startup
    with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
        index_html = f.read()

    async def index_handler(request):
        return web.Response(text=index_html, content_type="text/html")

    async def websocket_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        print("[WEB] Client connected.")

        # Clear any stale audio from previous sessions
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

        fps = args.fps

        async def send_frames():
            frame_interval = 1.0 / fps
            while not ws.closed and not stop_event.is_set():
                try:
                    try:
                        # Get pre-formatted bytes (audio or jpeg) from single queue
                        # This enforces sequential sending order (Audio -> Frames -> Audio -> Frames)
                        data = socket_queue.get(timeout=0.01)
                        await ws.send_bytes(data)
                        # Don't sleep if we have data, to clear backlog
                    except queue.Empty:
                        await asyncio.sleep(frame_interval)
                except (ConnectionResetError, ConnectionError):
                    break

        send_task = asyncio.create_task(send_frames())

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY:
                    chunk = np.frombuffer(msg.data, dtype=np.float32)
                    try:
                        audio_queue.put_nowait(chunk)
                    except queue.Full:
                        pass
                elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                    break
        finally:
            send_task.cancel()
            print("[WEB] Client disconnected.")

        return ws

    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", websocket_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()
    print(f"[WEB] Server running at http://0.0.0.0:{args.port} (IPv4)")
    print(f"[WEB] Open http://127.0.0.1:{args.port} in your browser.\n")

    try:
        while not stop_event.is_set():
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[WEB] Interrupted.")
    finally:
        stop_event.set()
        gen_thread.join(timeout=5)
        await runner.cleanup()
        print("[WEB] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
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
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--vad-threshold", type=float, default=0.005,
                        help="Server-side RMS silence threshold (0.0 = disabled). "
                             "Clips below this are discarded.")
    parser.add_argument("--use-init-latent", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable latent state preservation for continuity between clips. "
                             "Use --no-use-init-latent to disable (old behavior).")
    parser.add_argument("--audio-margin", type=int, default=2,
                        help="Audio feature context margin (frames). Higher = more context for lip sync.")
    args = parser.parse_args()

    setup_logging()

    pipe = load_pipeline(args.config, DEVICE, WEIGHT_DTYPE)

    assert os.path.exists(args.reference_image), f"Not found: {args.reference_image}"
    ref_image = Image.open(args.reference_image).convert("RGB")
    print(f"[INIT] Reference image: {args.reference_image}")

    assert os.path.isdir(args.pose_dir), f"Not found: {args.pose_dir}"
    pose_files = load_pose_files(args.pose_dir)
    print(f"[INIT] Loaded {len(pose_files)} pose files from {args.pose_dir}")

    with torch.no_grad():
        asyncio.run(run_server(pipe, ref_image, args.pose_dir, pose_files, args))


if __name__ == "__main__":
    main()
