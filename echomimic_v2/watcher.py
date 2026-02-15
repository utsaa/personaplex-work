"""Real-time EchoMimic-v2 watcher.

Captures microphone audio via sounddevice, feeds it through the accelerated
EchoMimic-v2 diffusion pipeline, and displays the generated talking-head
video in an OpenCV window — all in real-time (with ~1-2 s latency per clip).

Architecture (mirrors server.py decoupled queue pattern):
  ┌──────────┐     ┌──────────────┐     ┌──────────────┐
  │ audio_in │ ──► │ video_gen    │ ──► │ video_display│
  │ (sd)     │     │ (diffusion)  │     │ (cv2 window) │
  └──────────┘     └──────────────┘     └──────────────┘
   sd callback        Thread               Thread
   fills queue      drains audio_q        drains frame_q
                    fills  frame_q        shows  frames
"""

import argparse
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

# EchoMimic-v2 imports (local src/)
from diffusers import AutoencoderKL, DDIMScheduler
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2_acc import EchoMimicV2Pipeline
from src.models.pose_encoder import PoseEncoder
from src.utils.dwpose_util import draw_pose_select_v2

# ---------------------------------------------------------------------------
# Configuration defaults (overridable via CLI)
# ---------------------------------------------------------------------------
CONFIG_PATH = "./configs/prompts/infer_acc.yaml"
DEFAULT_REF_IMAGE = "./assets/therapist_ref.png"
DEFAULT_POSE_DIR = "./assets/halfbody_demo/pose/01"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DTYPE = torch.float16

# ---------------------------------------------------------------------------
# Model loading (same pattern as server.py / original watcher)
# ---------------------------------------------------------------------------

def load_pipeline(config_path: str, device: str, weight_dtype: torch.dtype) -> EchoMimicV2Pipeline:
    """Load all EchoMimic-v2 ACC models and return an assembled pipeline."""
    print("[INIT] Loading EchoMimic-v2 (ACC) models ...")
    config = OmegaConf.load(config_path)
    infer_config = OmegaConf.load(config.inference_config)

    # VAE
    print("  loading VAE ...")
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path, local_files_only=True, torch_dtype=weight_dtype,
    ).to(device=device, dtype=weight_dtype)

    # Reference UNet (2D)
    print("  loading reference UNet ...")
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path, subfolder="unet",
    ).to(device=device, dtype=weight_dtype)
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )

    # Denoising UNet (3D + motion module)
    print("  loading denoising UNet (ACC) ...")
    if os.path.exists(config.motion_module_path):
        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs),
        ).to(dtype=weight_dtype, device=device)
    else:
        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path, "",
            subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
                "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim,
            },
        ).to(dtype=weight_dtype, device=device)
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"), strict=False,
    )

    # Pose encoder
    print("  loading pose encoder ...")
    pose_net = PoseEncoder(
        320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256),
    ).to(device=device, dtype=weight_dtype)
    pose_net.load_state_dict(torch.load(config.pose_encoder_path, map_location="cpu"))

    # Audio processor (Whisper tiny)
    print("  loading audio processor (Whisper tiny) ...")
    audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

    # Scheduler
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    # Assemble pipeline
    pipe = EchoMimicV2Pipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        pose_encoder=pose_net,
        scheduler=scheduler,
    )
    pipe = pipe.to(device, dtype=weight_dtype)
    print("[READY] Pipeline loaded.\n")
    return pipe


# ---------------------------------------------------------------------------
# Pose helpers (same logic as server.py _prepare_pose_tensor)
# ---------------------------------------------------------------------------

def load_pose_files(pose_dir: str) -> list[str]:
    """Return sorted .npy filenames from pose_dir."""
    files = sorted(
        [f for f in os.listdir(pose_dir) if f.endswith(".npy")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    assert len(files) > 0, f"No .npy pose files found in {pose_dir}"
    return files


def prepare_pose_tensor(
    pose_dir: str,
    pose_files: list[str],
    num_frames: int,
    start_idx: int,
    W: int,
    H: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build (1, 3, L, H, W) pose tensor, cycling through available poses."""
    num_available = len(pose_files)
    pose_list = []
    for i in range(num_frames):
        idx = (start_idx + i) % num_available
        tgt_musk = np.zeros((W, H, 3), dtype=np.uint8)
        tgt_musk_path = os.path.join(pose_dir, pose_files[idx])
        detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
        imh_new, imw_new, rb, re, cb, ce = detected_pose["draw_pose_params"]
        im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
        im = np.transpose(np.array(im), (1, 2, 0))
        tgt_musk[rb:re, cb:ce, :] = im
        tgt_musk_pil = Image.fromarray(tgt_musk).convert("RGB")
        pose_list.append(
            torch.Tensor(np.array(tgt_musk_pil))
            .to(dtype=dtype, device=device)
            .permute(2, 0, 1)
            / 255.0
        )
    return torch.stack(pose_list, dim=1).unsqueeze(0)


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def generate_video_clip(
    pipe: EchoMimicV2Pipeline,
    ref_image: Image.Image,
    wav_path: str,
    poses_tensor: torch.Tensor,
    W: int, H: int,
    clip_frames: int,
    sample_rate: int,
    fps: int,
    steps: int = 4,
    cfg: float = 1.0,
) -> np.ndarray | None:
    """Run the ACC pipeline. Returns (1,3,L,H,W) numpy or None."""
    generator = torch.manual_seed(random.randint(0, 2**32 - 1))
    result = pipe(
        ref_image,
        wav_path,
        poses_tensor[:, :, :clip_frames, ...],
        W, H, clip_frames,
        steps, cfg,
        generator=generator,
        audio_sample_rate=sample_rate,
        context_frames=12,
        fps=fps,
        context_overlap=3,
        start_idx=0,
    )
    video = result.videos
    if isinstance(video, torch.Tensor):
        return video.cpu().numpy()
    return video


# ---------------------------------------------------------------------------
# Thread: audio capture (sounddevice callback -> audio_queue)
# ---------------------------------------------------------------------------

def audio_capture_thread(
    audio_queue: queue.Queue,
    sample_rate: int,
    stop_event: threading.Event,
    sd_device_index: int | None = None,
):
    """Continuously capture mic audio and push float32 chunks to audio_queue."""
    import sounddevice as sd

    block_duration = 0.1  # 100 ms blocks
    blocksize = int(sample_rate * block_duration)

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[AUDIO] {status}")
        # indata is (frames, channels) float32
        # put_nowait: callback runs on PortAudio's real-time thread and must
        # NEVER block, otherwise audio samples are silently dropped by the OS.
        # audio_queue is unbounded so Full should never happen, but guard anyway.
        try:
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # drop chunk rather than block the audio thread

    print(f"[AUDIO] Opening input stream @ {sample_rate} Hz ...")
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        device=sd_device_index,
        callback=callback,
    ):
        stop_event.wait()
    print("[AUDIO] Input stream closed.")


# ---------------------------------------------------------------------------
# Thread: video generation (audio_queue -> diffusion -> frame_queue)
# ---------------------------------------------------------------------------

def video_generation_thread(
    pipe: EchoMimicV2Pipeline,
    ref_image: Image.Image,
    pose_dir: str,
    pose_files: list[str],
    audio_queue: queue.Queue,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
    sample_rate: int = 16000,
    fps: int = 24,
    clip_frames: int = 12,
    W: int = 512,
    H: int = 512,
    steps: int = 4,
    cfg: float = 1.0,
):
    """Accumulate audio -> temp WAV -> pipeline -> push BGR frames to frame_queue."""
    samples_per_clip = int(sample_rate * clip_frames / fps)
    audio_buffer = np.array([], dtype=np.float32)
    pose_idx = 0

    print(f"[GEN] Waiting for audio (need {samples_per_clip} samples = "
          f"{clip_frames/fps:.2f}s per clip) ...")

    while not stop_event.is_set():
        # Drain audio_queue into buffer
        try:
            while True:
                chunk = audio_queue.get(timeout=0.05)
                audio_buffer = np.concatenate((audio_buffer, chunk))
        except queue.Empty:
            pass

        # Not enough audio yet
        if len(audio_buffer) < samples_per_clip:
            continue

        # Take one clip's worth
        clip_audio = audio_buffer[:samples_per_clip]
        audio_buffer = audio_buffer[samples_per_clip:]

        tmp_wav_path = None
        try:
            # Write temp WAV (Whisper expects a file path)
            tmp_fd, tmp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(tmp_fd)
            with wave.open(tmp_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                pcm_int16 = np.clip(clip_audio * 32767, -32768, 32767).astype(np.int16)
                wf.writeframes(pcm_int16.tobytes())

            # Prepare pose tensor
            poses_tensor = prepare_pose_tensor(
                pose_dir, pose_files, clip_frames, pose_idx, W, H,
                device=DEVICE, dtype=WEIGHT_DTYPE,
            )
            pose_idx = (pose_idx + clip_frames) % len(pose_files)

            # Run diffusion
            t0 = time.perf_counter()
            video_np = generate_video_clip(
                pipe, ref_image, tmp_wav_path, poses_tensor,
                W, H, clip_frames, sample_rate, fps, steps, cfg,
            )
            dt = time.perf_counter() - t0
            print(f"[GEN] Clip generated in {dt:.2f}s "
                  f"({clip_frames} frames, {clip_frames/fps:.2f}s of video)")

            # Push individual BGR frames into frame_queue
            if video_np is not None:
                n_frames = video_np.shape[2]
                for f_idx in range(n_frames):
                    if stop_event.is_set():
                        return
                    frame = video_np[0, :, f_idx, :, :]  # (3, H, W) in [0,1]
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                    frame = frame.transpose(1, 2, 0)  # (H, W, 3) RGB
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    try:
                        frame_queue.put_nowait(frame_bgr)
                    except queue.Full:
                        pass  # drop if display can't keep up

        except Exception as e:
            print(f"[GEN] Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            if tmp_wav_path is not None:
                try:
                    os.unlink(tmp_wav_path)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Main thread: video display (frame_queue -> cv2.imshow)
# ---------------------------------------------------------------------------

def display_loop(
    frame_queue: queue.Queue,
    stop_event: threading.Event,
    fps: int = 24,
    window_name: str = "EchoMimic-v2 Live",
):
    """Pop frames from frame_queue and display at target FPS. Runs on main thread."""
    frame_interval = 1.0 / fps
    last_frame = None

    print(f"[DISPLAY] Showing video at {fps} FPS. Press 'q' or ESC to quit.\n")

    while not stop_event.is_set():
        try:
            frame_bgr = frame_queue.get(timeout=0.1)
            last_frame = frame_bgr
        except queue.Empty:
            pass  # no new frame -- keep showing last one (or blank)

        if last_frame is not None:
            cv2.imshow(window_name, last_frame)

        key = cv2.waitKey(max(1, int(frame_interval * 1000)))
        if key in (ord("q"), 27):  # 'q' or ESC
            stop_event.set()
            break

    cv2.destroyAllWindows()
    print("[DISPLAY] Window closed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real-time EchoMimic-v2 watcher (sd audio -> video)")
    parser.add_argument("--config", type=str, default=CONFIG_PATH,
                        help="Path to infer_acc.yaml config.")
    parser.add_argument("--reference-image", type=str, default=DEFAULT_REF_IMAGE,
                        help="Reference face image for animation.")
    parser.add_argument("--pose-dir", type=str, default=DEFAULT_POSE_DIR,
                        help="Directory with pose .npy files.")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Audio sample rate for capture & Whisper (default: 16000).")
    parser.add_argument("--fps", type=int, default=24, help="Video FPS (default: 24).")
    parser.add_argument("--clip-frames", type=int, default=12,
                        help="Frames per clip (default: 12 = 0.5s at 24fps).")
    parser.add_argument("--width", type=int, default=512, help="Video width (default: 512).")
    parser.add_argument("--height", type=int, default=512, help="Video height (default: 512).")
    parser.add_argument("--steps", type=int, default=4,
                        help="Inference steps (ACC default: 4).")
    parser.add_argument("--cfg", type=float, default=1.0, help="Guidance scale (default: 1.0).")
    parser.add_argument("--sd-device", type=int, default=None,
                        help="Sounddevice input device index (None = system default).")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio devices and exit.")
    args = parser.parse_args()

    # Optionally just list audio devices
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    # Load pipeline
    pipe = load_pipeline(args.config, DEVICE, WEIGHT_DTYPE)

    # Load reference image
    assert os.path.exists(args.reference_image), f"Reference image not found: {args.reference_image}"
    ref_image = Image.open(args.reference_image).convert("RGB")
    print(f"[INIT] Reference image: {args.reference_image}")

    # Load pose files
    assert os.path.isdir(args.pose_dir), f"Pose directory not found: {args.pose_dir}"
    pose_files = load_pose_files(args.pose_dir)
    print(f"[INIT] Loaded {len(pose_files)} pose files from {args.pose_dir}")

    # Shared queues & stop event
    audio_queue = queue.Queue()                # unbounded – sd callback must NEVER block
    frame_queue = queue.Queue(maxsize=200)     # BGR frames ready for display
    stop_event = threading.Event()

    # Start audio capture thread
    audio_thread = threading.Thread(
        target=audio_capture_thread,
        args=(audio_queue, args.sample_rate, stop_event, args.sd_device),
        daemon=True,
    )
    audio_thread.start()

    # Start video generation thread
    gen_thread = threading.Thread(
        target=video_generation_thread,
        args=(
            pipe, ref_image, args.pose_dir, pose_files,
            audio_queue, frame_queue, stop_event,
            args.sample_rate, args.fps, args.clip_frames,
            args.width, args.height, args.steps, args.cfg,
        ),
        daemon=True,
    )
    gen_thread.start()

    # Run display on main thread (cv2.imshow requires main thread on most OS)
    try:
        display_loop(frame_queue, stop_event, fps=args.fps)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted.")
    finally:
        stop_event.set()
        audio_thread.join(timeout=2)
        gen_thread.join(timeout=5)
        print("[MAIN] Done.")


if __name__ == "__main__":
    with torch.no_grad():
        main()
