"""Real-time EchoMimic-v2 watcher (Multi-GPU).

Captures microphone audio via sounddevice, feeds it through the accelerated
EchoMimic-v2 diffusion pipeline, and displays the generated talking-head
video in an OpenCV window — all in real-time.

Supports 1..N CUDA GPUs:
  - N == 1: sequential generation with init_latent continuity
  - N >= 2: pipelined overlap-blend for parallel GPU usage
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

WEIGHT_DTYPE = torch.float16

# ---------------------------------------------------------------------------
# GPU Detection & Utilities
# ---------------------------------------------------------------------------

def detect_gpus() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def blend_overlap(tail_video: np.ndarray, head_video: np.ndarray, K: int) -> np.ndarray:
    """Linear crossfade: tail[-K:] × head[:K]. Shape (1,C,K,H,W)."""
    if K <= 0:
        return np.empty((tail_video.shape[0], tail_video.shape[1], 0,
                         tail_video.shape[3], tail_video.shape[4]),
                        dtype=tail_video.dtype)
    tail_slice = tail_video[:, :, -K:, :, :]
    head_slice = head_video[:, :, :K, :, :]
    alpha = np.linspace(0.0, 1.0, K, dtype=np.float32).reshape(1, 1, K, 1, 1)
    return (1.0 - alpha) * tail_slice + alpha * head_slice


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pipeline(config_path: str, device: str, weight_dtype: torch.dtype) -> EchoMimicV2Pipeline:
    """Load all EchoMimic-v2 ACC models on a specific device."""
    print(f"[INIT] Loading EchoMimic-v2 (ACC) models on {device} ...")
    config = OmegaConf.load(config_path)
    infer_config = OmegaConf.load(config.inference_config)

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path, local_files_only=True, torch_dtype=weight_dtype,
    ).to(device=device, dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path, subfolder="unet",
    ).to(device=device, dtype=weight_dtype)
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )

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

    pose_net = PoseEncoder(
        320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256),
    ).to(device=device, dtype=weight_dtype)
    pose_net.load_state_dict(torch.load(config.pose_encoder_path, map_location="cpu"))

    audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    pipe = EchoMimicV2Pipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        pose_encoder=pose_net,
        scheduler=scheduler,
    )
    pipe = pipe.to(device, dtype=weight_dtype)
    print(f"[READY] Pipeline on {device} loaded.\n")
    return pipe


def load_all_pipelines(config_path: str, weight_dtype: torch.dtype):
    """Load N pipelines, one per GPU. Returns (pipes, devices, num_gpus)."""
    N = max(1, detect_gpus())
    if N == 0:
        devices = ["cpu"]
        N = 1
    else:
        devices = [f"cuda:{i}" for i in range(N)]

    pipes = [None] * N
    errors = [None] * N

    if N == 1:
        pipes[0] = load_pipeline(config_path, devices[0], weight_dtype)
    else:
        def _load(idx):
            try:
                pipes[idx] = load_pipeline(config_path, devices[idx], weight_dtype)
            except Exception as e:
                errors[idx] = e

        threads = []
        for i in range(N):
            t = threading.Thread(target=_load, args=(i,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        for i, err in enumerate(errors):
            if err is not None:
                raise RuntimeError(f"Failed to load on {devices[i]}: {err}")

    print(f"[INIT] {N} pipeline(s) loaded.")
    return pipes, devices, N


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
    pose_dir, pose_files, num_frames, start_idx, W, H, device, dtype,
) -> torch.Tensor:
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
) -> tuple:
    """Run pipeline. Returns (video_np, final_latent)."""
    generator = torch.manual_seed(random.randint(0, 2**32 - 1))
    result = pipe(
        ref_image, wav_path,
        poses_tensor[:, :, :clip_frames, ...],
        W, H, clip_frames,
        steps, cfg,
        generator=generator,
        audio_sample_rate=sample_rate,
        context_frames=12, fps=fps,
        context_overlap=3, start_idx=0,
        init_latents=init_latent if use_init_latent else None,
    )
    video = result.videos
    final_latent = getattr(result, 'final_latent', None)
    if isinstance(video, torch.Tensor):
        return video.cpu().numpy(), final_latent
    return video, final_latent


# ---------------------------------------------------------------------------
# Thread: audio capture (sounddevice callback -> audio_queue)
# ---------------------------------------------------------------------------

def audio_capture_thread(
    audio_queue, sample_rate, stop_event, sd_device_index=None,
):
    import sounddevice as sd
    block_duration = 0.1
    blocksize = int(sample_rate * block_duration)

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[AUDIO] {status}")
        try:
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    print(f"[AUDIO] Opening input stream @ {sample_rate} Hz ...")
    with sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="float32",
        blocksize=blocksize, device=sd_device_index, callback=callback,
    ):
        stop_event.wait()
    print("[AUDIO] Input stream closed.")


# ---------------------------------------------------------------------------
# Thread: video generation — unified 1..N GPU
# ---------------------------------------------------------------------------

def video_generation_thread(
    pipes, devices, num_gpus, overlap_frames,
    ref_image, pose_dir, pose_files,
    audio_queue, frame_queue, stop_event,
    sample_rate=16000, fps=24, clip_frames=12,
    W=512, H=512, steps=4, cfg=1.0,
):
    """Generate video clips, alternating GPUs with overlap-blend for N>=2."""
    samples_per_clip = int(sample_rate * clip_frames / fps)
    audio_buffer = np.array([], dtype=np.float32)
    pose_idx = 0
    
    K = overlap_frames if num_gpus >= 2 else 0
    gpu_idx = 0
    is_first_chunk = True
    
    # Per-GPU state
    last_latents = [None] * num_gpus  # For single-GPU init_latent continuity
    tail_buffers = [None] * num_gpus  # For multi-GPU overlap blending

    print(f"[GEN] Waiting for audio (need {samples_per_clip} samples = "
          f"{clip_frames/fps:.2f}s per clip), {num_gpus} GPU(s), K={K} ...")

    while not stop_event.is_set():
        try:
            while True:
                chunk = audio_queue.get(timeout=0.05)
                audio_buffer = np.concatenate((audio_buffer, chunk))
        except queue.Empty:
            pass

        if len(audio_buffer) < samples_per_clip:
            continue

        clip_audio = audio_buffer[:samples_per_clip]
        audio_buffer = audio_buffer[samples_per_clip:]

        tmp_wav_path = None
        try:
            tmp_fd, tmp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(tmp_fd)
            with wave.open(tmp_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                pcm_int16 = np.clip(clip_audio * 32767, -32768, 32767).astype(np.int16)
                wf.writeframes(pcm_int16.tobytes())

            device = devices[gpu_idx]
            pipe = pipes[gpu_idx]

            gen_frames = clip_frames + K if is_first_chunk else K + clip_frames
            poses_tensor = prepare_pose_tensor(
                pose_dir, pose_files, gen_frames, pose_idx, W, H,
                device=device, dtype=WEIGHT_DTYPE,
            )
            pose_idx = (pose_idx + clip_frames) % len(pose_files)

            t0 = time.perf_counter()

            if num_gpus >= 2:
                # Multi-GPU: overlap-blend mode
                video_np, _ = generate_video_clip(
                    pipe, ref_image, tmp_wav_path, poses_tensor,
                    W, H, gen_frames, sample_rate, fps, steps, cfg,
                    init_latent=None, use_init_latent=False,
                )

                if video_np is not None:
                    if is_first_chunk:
                        output_video = video_np
                        tail_buffers[gpu_idx] = video_np[:, :, -K:, :, :].copy() if K > 0 else None
                        is_first_chunk = False
                    else:
                        prev_gpu = (gpu_idx - 1) % num_gpus
                        tail_buf = tail_buffers[prev_gpu]
                        tail_buffers[prev_gpu] = None

                        if tail_buf is not None and K > 0:
                            blended = blend_overlap(tail_buf, video_np, K)
                            output_video = np.concatenate([blended, video_np[:, :, K:, :, :]], axis=2)
                            output_video = output_video[:, :, -clip_frames:, :, :]
                        else:
                            output_video = video_np[:, :, -clip_frames:, :, :]

                        tail_buffers[gpu_idx] = video_np[:, :, -K:, :, :].copy() if K > 0 else None

                    gpu_idx = (gpu_idx + 1) % num_gpus
            else:
                # Single-GPU: init_latent continuity
                video_np, final_latent = generate_video_clip(
                    pipe, ref_image, tmp_wav_path, poses_tensor,
                    W, H, clip_frames, sample_rate, fps, steps, cfg,
                    init_latent=last_latents[0], use_init_latent=True,
                )
                last_latents[0] = final_latent
                output_video = video_np

            dt = time.perf_counter() - t0
            print(f"[GEN] GPU {gpu_idx}: Clip in {dt:.2f}s "
                  f"({gen_frames} gen, {output_video.shape[2] if output_video is not None else 0} out)")

            if output_video is not None:
                n_frames = output_video.shape[2]
                for f_idx in range(n_frames):
                    if stop_event.is_set():
                        return
                    frame = output_video[0, :, f_idx, :, :]
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                    frame = frame.transpose(1, 2, 0)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    try:
                        frame_queue.put_nowait(frame_bgr)
                    except queue.Full:
                        pass

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
# Main thread: video display
# ---------------------------------------------------------------------------

def display_loop(frame_queue, stop_event, fps=24, window_name="EchoMimic-v2 Live"):
    frame_interval = 1.0 / fps
    last_frame = None
    print(f"[DISPLAY] Showing video at {fps} FPS. Press 'q' or ESC to quit.\n")

    while not stop_event.is_set():
        try:
            frame_bgr = frame_queue.get(timeout=0.1)
            last_frame = frame_bgr
        except queue.Empty:
            pass

        if last_frame is not None:
            cv2.imshow(window_name, last_frame)

        key = cv2.waitKey(max(1, int(frame_interval * 1000)))
        if key in (ord("q"), 27):
            stop_event.set()
            break

    cv2.destroyAllWindows()
    print("[DISPLAY] Window closed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real-time EchoMimic-v2 watcher (multi-GPU)")
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument("--reference-image", type=str, default=DEFAULT_REF_IMAGE)
    parser.add_argument("--pose-dir", type=str, default=DEFAULT_POSE_DIR)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--clip-frames", type=int, default=12)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--sd-device", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--overlap-frames", type=int, default=6,
                        help="Overlap frames (K) for multi-GPU blending. Default: 6.")
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    # Load N pipelines
    pipes, devices, num_gpus = load_all_pipelines(args.config, WEIGHT_DTYPE)
    K = args.overlap_frames if num_gpus >= 2 else 0
    print(f"[INIT] {num_gpus} GPU(s), overlap K={K}")

    # Reference image
    assert os.path.exists(args.reference_image), f"Not found: {args.reference_image}"
    ref_image = Image.open(args.reference_image).convert("RGB")
    print(f"[INIT] Reference image: {args.reference_image}")

    # Pose files
    assert os.path.isdir(args.pose_dir), f"Not found: {args.pose_dir}"
    pose_files = load_pose_files(args.pose_dir)
    print(f"[INIT] Loaded {len(pose_files)} pose files from {args.pose_dir}")

    # Queues & stop event
    audio_queue = queue.Queue()
    frame_queue = queue.Queue(maxsize=200)
    stop_event = threading.Event()

    # Audio capture thread
    audio_thread = threading.Thread(
        target=audio_capture_thread,
        args=(audio_queue, args.sample_rate, stop_event, args.sd_device),
        daemon=True,
    )
    audio_thread.start()

    # Video generation thread (multi-GPU aware)
    gen_thread = threading.Thread(
        target=video_generation_thread,
        args=(
            pipes, devices, num_gpus, K,
            ref_image, args.pose_dir, pose_files,
            audio_queue, frame_queue, stop_event,
            args.sample_rate, args.fps, args.clip_frames,
            args.width, args.height, args.steps, args.cfg,
        ),
        daemon=True,
    )
    gen_thread.start()

    # Display on main thread
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
