
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np
import torch
import random
from PIL import Image

from core.audio import _rms
from core.gpu import MultiGPUManager, blend_overlap
from core.pose import PoseProvider

# Maximum time to wait for a single GPU generation result (seconds)
GPU_RESULT_TIMEOUT_S: int = 300

# ---------------------------------------------------------------------------
# Worker 1: Input Preparation (CPU)
# ---------------------------------------------------------------------------

def input_preparation_thread(
    audio_queue: queue.Queue,
    prepared_queue: queue.Queue,
    stop_event: threading.Event,
    pose_provider: PoseProvider,
    sample_rate: int,
    fps: int,
    clip_frames: int,
    vad_threshold: float = 0.005,
    audio_margin: int = 2,
    overlap_frames: int = 0,
) -> None:
    """
    Consumes raw audio chunks, performs VAD, prepares WAV/Pose inputs,
    and pushes `(audio_data, poses_tensor, ctx_frames, audio_chunk, reset_latent)` 
    to `prepared_queue`.
    """
    samples_per_clip = int(sample_rate * clip_frames / fps)
    
    # Audio Context Buffer: 2.0s total context budget.
    # For multi-GPU (overlap_frames > 0), shortened by K so inserting
    # K prev-audio frames keeps total context at exactly 2.0s.
    k_samples = int(sample_rate * overlap_frames / fps)
    history_samples = sample_rate * 2 - k_samples
    audio_history_buffer = np.zeros(history_samples, dtype=np.float32)
    
    # Incoming accumulation buffer
    incoming_audio_buffer = np.array([], dtype=np.float32)

    pose_idx = 0
    consecutive_silent_clips = 0
    idle_clips_limit = max(1, int(1.0 * fps / clip_frames))

    print(f"[PREP] Waiting for audio (target clip: {samples_per_clip} samples) ...")
    print(f"[PREP] Audio Context: {history_samples/sample_rate:.2f}s history + "
          f"{clip_frames/fps:.1f}s new (overlap_frames={overlap_frames})")
    print(f"[PREP] Silence timeout: 1.0s (stop updating after {idle_clips_limit} silent clips)")
    
    while not stop_event.is_set():
        # 1. Drain input queue
        try:
            while True:
                chunk = audio_queue.get_nowait()
                incoming_audio_buffer = np.concatenate((incoming_audio_buffer, chunk))
        except queue.Empty:
            pass

        # 2. Process clips
        if len(incoming_audio_buffer) >= samples_per_clip:
            current_audio = incoming_audio_buffer[:samples_per_clip]
            incoming_audio_buffer = incoming_audio_buffer[samples_per_clip:]

            # Silence Gate
            clip_rms = _rms(current_audio)
            reset_latent = False

            if vad_threshold > 0 and clip_rms < vad_threshold:
                consecutive_silent_clips += 1
                if consecutive_silent_clips <= idle_clips_limit:
                    # Keep updating history for smooth fade-out
                    audio_history_buffer = np.concatenate((audio_history_buffer, current_audio))[-history_samples:]
                elif consecutive_silent_clips % (idle_clips_limit * 6) == 0:
                    print(f"[PREP] System idle... (Silent for {consecutive_silent_clips})")
                
                # If silent for too long, signal next generation to reset latent
                if consecutive_silent_clips > (idle_clips_limit * 4):
                    pass
                continue

            # Speech detected
            if consecutive_silent_clips > 0:
                print(f"[PREP] Audio detected (RMS={clip_rms:.4f}). Starting...")
                # If we were silent for a long time, the GPU thread's last_latent is stale.
                if consecutive_silent_clips > (idle_clips_limit * 4):
                    reset_latent = True
            
            consecutive_silent_clips = 0

            # Prepare Inputs
            # Create full window for model context: 2.0s history + current + (optional) silence
            # We ONLY add silence padding if the speech is ending (low RMS).
            # This prevents the mouth from closing unnaturally between syllables.
            if clip_rms < vad_threshold * 1.5:
                post_padding_samples = int(sample_rate * 0.2)
                post_padding = np.zeros(post_padding_samples, dtype=np.float32)
            else:
                post_padding = np.array([], dtype=np.float32)
            
            full_window = np.concatenate((audio_history_buffer, current_audio, post_padding))
            ctx_duration = len(audio_history_buffer) / sample_rate
            ctx_frames = int(ctx_duration * fps)

            try:
                # Prepare Pose Tensor using Provider
                poses_tensor = pose_provider.get_batch(pose_idx, clip_frames)
                pose_idx = pose_idx + clip_frames

                # Push to GPU thread
                batch = (full_window, poses_tensor, ctx_frames, current_audio.copy(), reset_latent)
                prepared_queue.put(batch)

                # Update history
                audio_history_buffer = np.concatenate((audio_history_buffer, current_audio))[-history_samples:]

            except Exception as e:
                print(f"[PREP] Error: {e}")
                pass
        else:
            # Wait a bit if not enough audio
            time.sleep(0.005)


# ---------------------------------------------------------------------------
# Worker 2: Video Generation (GPU) — Unified for 1..N GPUs
# ---------------------------------------------------------------------------

def generate_video_clip(
    pipe: object,
    ref_image: Image.Image,
    audio_data: np.ndarray,
    poses_tensor: torch.Tensor,
    W: int,
    H: int,
    clip_frames: int,
    sample_rate: int,
    fps: int,
    steps: int = 4,
    cfg: float = 1.0,
    init_latent: Optional[torch.Tensor] = None,
    use_init_latent: bool = True,
    audio_margin: int = 2,
    reference_cache: Optional[object] = None,
    audio_context_frames: int = 0,
) -> tuple[Optional[np.ndarray], Optional[torch.Tensor]]:
    """Wraps pipeline call."""
    generator = torch.manual_seed(random.randint(0, 2**32 - 1))
    result = pipe(
        ref_image, audio_data,
        poses_tensor[:, :, :clip_frames, ...],
        W, H, clip_frames, steps, cfg,
        generator=generator,
        audio_sample_rate=sample_rate,
        context_frames=12, fps=fps,
        context_overlap=3, start_idx=0,
        audio_margin=audio_margin, 
        init_latents=init_latent if use_init_latent else None,
        reference_cache=reference_cache,
        audio_context_frames=audio_context_frames,
    )
    video = result.videos
    final_latent = result.final_latent
    
    if isinstance(video, torch.Tensor):
        video_np = video.cpu().numpy()
    else:
        video_np = video
    
    return video_np, final_latent


def video_generation_thread(
    gpu_manager: MultiGPUManager,
    ref_image: Image.Image,
    prepared_queue: queue.Queue,
    raw_clip_queue: queue.Queue,
    stop_event: threading.Event,
    sample_rate: int,
    fps: int,
    clip_frames: int,
    W: int,
    H: int,
    steps: int,
    cfg: float,
    use_init_latent: bool = True,
    audio_margin: int = 2,
) -> None:
    """
    Unified video generation thread for 1..N GPUs.
    
    - N == 1: Sequential generation with init_latent continuity (old behavior).
    - N >= 2: TRUE PARALLEL generation — submits one clip per GPU simultaneously
      via ThreadPoolExecutor, then collects results IN ORDER for correct blending.
      
    Ordering guarantee: chunks are always blended and sent to `raw_clip_queue`
    in the same order they were submitted.  GPU 0's result is processed before
    GPU 1's, even if GPU 1 finishes first.
    """
    num_gpus = gpu_manager.num_gpus
    is_multi = gpu_manager.is_multi_gpu
    K = gpu_manager.overlap_frames  # 0 for single GPU, 6 for multi

    is_first_chunk = True  # After silence reset or startup

    # Tail buffer for multi-GPU overlap blending (keeps last K frames)
    tail_buffer: Optional[np.ndarray] = None

    print(f"[GEN] Video generation started — {num_gpus} GPU(s), "
          f"overlap K={K}, clip_frames={clip_frames}")

    if is_multi:
        _run_multi_gpu_loop(
            gpu_manager, ref_image, prepared_queue, raw_clip_queue, stop_event,
            sample_rate, fps, clip_frames, W, H, steps, cfg, audio_margin,
            num_gpus, K,
        )
    else:
        _run_single_gpu_loop(
            gpu_manager, ref_image, prepared_queue, raw_clip_queue, stop_event,
            sample_rate, fps, clip_frames, W, H, steps, cfg,
            use_init_latent, audio_margin,
        )


def _run_single_gpu_loop(
    gpu_manager: MultiGPUManager,
    ref_image: Image.Image,
    prepared_queue: queue.Queue,
    raw_clip_queue: queue.Queue,
    stop_event: threading.Event,
    sample_rate: int,
    fps: int,
    clip_frames: int,
    W: int,
    H: int,
    steps: int,
    cfg: float,
    use_init_latent: bool,
    audio_margin: int,
) -> None:
    """Single-GPU: sequential generation with init_latent continuity."""
    pipe, device = gpu_manager.get_pipeline(0)
    ref_cache = gpu_manager.reference_caches[0]

    while not stop_event.is_set():
        try:
            batch = prepared_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.002)
            continue

        audio_data, poses_tensor, ctx_frames, current_audio, reset_latent = batch

        if reset_latent:
            gpu_manager.last_latents[0] = None
            print("[GEN] Pipeline reset due to silence.")

        try:
            t0 = time.perf_counter()
            init_latent = gpu_manager.last_latents[0]

            video_np, final_latent = generate_video_clip(
                pipe, ref_image, audio_data, poses_tensor,
                W, H, clip_frames, sample_rate, fps, steps, cfg,
                init_latent=init_latent,
                use_init_latent=use_init_latent,
                audio_margin=audio_margin,
                reference_cache=ref_cache,
                audio_context_frames=ctx_frames,
            )

            if use_init_latent:
                gpu_manager.last_latents[0] = final_latent

            dt = time.perf_counter() - t0
            print(f"[GEN] Clip generated in {dt:.2f}s")

            if video_np is not None:
                raw_clip_queue.put((video_np, current_audio))

        except Exception as e:
            print(f"[GEN] Error: {e}")
            import traceback; traceback.print_exc()


def _gpu_generate_task(
    gpu_idx: int,
    gpu_manager: MultiGPUManager,
    ref_image: Image.Image,
    audio_data: np.ndarray,
    poses_tensor: torch.Tensor,
    gen_frames: int,
    W: int,
    H: int,
    sample_rate: int,
    fps: int,
    steps: int,
    cfg: float,
    audio_margin: int,
    ctx_frames: int,
) -> Optional[np.ndarray]:
    """Target function for ThreadPoolExecutor — runs on one GPU, returns result.
    
    This function is thread-safe: it only reads from gpu_manager (pipe, ref_cache)
    and does not mutate any shared state.  All mutable state (tail_buffer) is
    handled by the caller AFTER this function returns.
    """
    pipe, device = gpu_manager.get_pipeline(gpu_idx)
    ref_cache = gpu_manager.reference_caches[gpu_idx % gpu_manager.num_gpus]

    # Extend poses if shorter than gen_frames
    actual_pose_frames = poses_tensor.shape[2]
    if actual_pose_frames < gen_frames:
        pad = poses_tensor[:, :, -1:, :, :].repeat(
            1, 1, gen_frames - actual_pose_frames, 1, 1
        )
        poses_tensor = torch.cat([poses_tensor, pad], dim=2)

    video_np, _ = generate_video_clip(
        pipe, ref_image, audio_data,
        poses_tensor[:, :, :gen_frames, ...],
        W=W, H=H,
        clip_frames=gen_frames, sample_rate=sample_rate, fps=fps,
        steps=steps, cfg=cfg,
        init_latent=None,      # Multi-GPU: start from noise
        use_init_latent=False,
        audio_margin=audio_margin,
        reference_cache=ref_cache,
        audio_context_frames=ctx_frames,
    )
    return video_np


def _run_multi_gpu_loop(
    gpu_manager: MultiGPUManager,
    ref_image: Image.Image,
    prepared_queue: queue.Queue,
    raw_clip_queue: queue.Queue,
    stop_event: threading.Event,
    sample_rate: int,
    fps: int,
    clip_frames: int,
    W: int,
    H: int,
    steps: int,
    cfg: float,
    audio_margin: int,
    num_gpus: int,
    K: int,
) -> None:
    """Multi-GPU: fire-and-forget dispatch with ordered output.

    Strategy:
    - As soon as a batch arrives, dispatch it to the next GPU immediately.
    - Maintain an ordered deque of pending futures.
    - Process the HEAD of the deque when it completes — guarantees ordering.
    - Store the last K frames as tail_buffer for blending with the next chunk.
    
    Audio handling:
    - Each batch from input_preparation has audio for `clip_frames` (12) frames.
    - Multi-GPU generates `clip_frames + K` (18) frames per chunk.
    - We store each batch's `current_audio`.  For subsequent chunks, the last
      K frames of the PREVIOUS batch's audio are inserted between the history
      and current audio: [history(2s)][prev_K_audio][current(12)] = full
      context + 18 real audio frames.  ctx_frames stays unchanged.
    - First chunk generates only clip_frames (nothing to blend with yet).
    
    This keeps all GPUs busy without waiting to collect N batches first.
    """
    is_first_chunk = True
    tail_buffer: Optional[np.ndarray] = None  # Last K video frames from previous chunk
    prev_current_audio: Optional[np.ndarray] = None  # Previous batch's raw audio
    next_gpu = 0  # Round-robin counter

    samples_per_clip = int(sample_rate * clip_frames / fps)
    k_samples = int(sample_rate * K / fps)

    executor = ThreadPoolExecutor(max_workers=num_gpus)
    # Ordered queue of pending results
    pending: deque = deque()

    print(f"[GEN] Multi-GPU loop: {num_gpus} GPUs, K={K}, fire-and-forget dispatch")

    try:
        while not stop_event.is_set():
            # ---- Dispatch: grab any available batch and submit immediately ----
            try:
                batch = prepared_queue.get_nowait()
            except queue.Empty:
                batch = None

            if batch is not None:
                audio_data, poses_tensor, ctx_frames, current_audio, reset_latent = batch

                if reset_latent:
                    # Wait for all pending futures to finish before resetting
                    _drain_pending(pending, raw_clip_queue, tail_buffer,
                                   is_first_chunk, K, clip_frames)
                    is_first_chunk = True
                    tail_buffer = None
                    prev_current_audio = None
                    pending.clear()
                    print("[GEN] Pipeline reset due to silence.")

                gpu_idx = next_gpu % num_gpus

                if is_first_chunk:
                    # First chunk: no overlap needed (nothing to blend with).
                    # Generate exactly clip_frames. Save tail K for next chunk.
                    gen_frames = clip_frames
                    final_audio = audio_data
                    final_ctx = ctx_frames
                else:
                    # Subsequent chunks: generate clip_frames + K (e.g. 18).
                    # audio_data = [history (2s)][current_audio (12 frames)]
                    # We insert K frames of PREVIOUS audio between them:
                    # extended  = [history (2s)][prev_k_audio (K frames)][current_audio (12 frames)]
                    # This gives the pipeline full context + 18 frames of real audio.
                    gen_frames = clip_frames + K

                    # Get K frames of audio from previous batch
                    if prev_current_audio is not None and len(prev_current_audio) >= k_samples:
                        k_audio = prev_current_audio[-k_samples:]
                    else:
                        k_audio = np.zeros(k_samples, dtype=np.float32)

                    # Split audio_data into history and current parts
                    if len(audio_data) > samples_per_clip:
                        history_part = audio_data[:-samples_per_clip]
                        current_part = audio_data[-samples_per_clip:]
                    else:
                        history_part = np.array([], dtype=np.float32)
                        current_part = audio_data

                    # Reconstruct: [history][k_prev_audio][current] = ctx + K + 12 frames
                    final_audio = np.concatenate([history_part, k_audio, current_part])
                    final_ctx = ctx_frames  # Full context preserved

                # Save current audio for next iteration's K overlap
                prev_current_audio = current_audio.copy()

                future = executor.submit(
                    _gpu_generate_task,
                    gpu_idx, gpu_manager, ref_image,
                    final_audio, poses_tensor,
                    gen_frames, W, H, sample_rate, fps, steps, cfg, audio_margin,
                    final_ctx,
                )
                pending.append((future, current_audio, gen_frames, gpu_idx, is_first_chunk))

                # After first dispatch, subsequent chunks are no longer "first"
                if is_first_chunk:
                    is_first_chunk = False

                next_gpu += 1

            # ---- Process: check if the HEAD future is done (in-order) ----
            while pending and not stop_event.is_set():
                head_future, head_audio, head_gen_frames, head_gpu, head_was_first = pending[0]

                if not head_future.done():
                    break  # Head not ready yet — don't skip ahead

                pending.popleft()

                try:
                    video_np = head_future.result(timeout=GPU_RESULT_TIMEOUT_S)
                except Exception as e:
                    print(f"[GEN] GPU {head_gpu} failed: {e}")
                    import traceback; traceback.print_exc()
                    continue

                if video_np is None:
                    continue

                # ---- Blend and produce output ----
                if head_was_first:
                    # First chunk: send all frames, save tail
                    output_video = video_np
                    if K > 0 and video_np.shape[2] >= K:
                        tail_buffer = video_np[:, :, -K:, :, :].copy()
                else:
                    # Subsequent chunks: blend first K with previous tail
                    if tail_buffer is not None and K > 0:
                        blended = blend_overlap(tail_buffer, video_np, K)
                        output_video = np.concatenate([
                            blended,
                            video_np[:, :, K:, :, :]
                        ], axis=2)
                        # Only send the last clip_frames
                        output_video = output_video[:, :, -clip_frames:, :, :]
                    else:
                        output_video = video_np[:, :, -clip_frames:, :, :]

                    # Update tail buffer for next chunk
                    if K > 0 and video_np.shape[2] >= K:
                        tail_buffer = video_np[:, :, -K:, :, :].copy()

                print(f"[GEN] GPU {head_gpu}: "
                      f"{head_gen_frames} gen → {output_video.shape[2]} sent")

                raw_clip_queue.put((output_video, head_audio))

            # Small sleep only if nothing was dispatched AND nothing was processed
            if batch is None and (not pending or not pending[0][0].done()):
                time.sleep(0.002)

    finally:
        executor.shutdown(wait=False)


def _drain_pending(
    pending: deque,
    raw_clip_queue: queue.Queue,
    tail_buffer: Optional[np.ndarray],
    is_first_chunk: bool,
    K: int,
    clip_frames: int,
) -> None:
    """Wait for and process all pending futures before a reset."""
    while pending:
        future, audio, gen_frames, gpu_idx, was_first = pending.popleft()
        try:
            video_np = future.result(timeout=GPU_RESULT_TIMEOUT_S)
            if video_np is None:
                continue

            if was_first:
                output_video = video_np
                if K > 0 and video_np.shape[2] >= K:
                    tail_buffer = video_np[:, :, -K:, :, :].copy()
            else:
                if tail_buffer is not None and K > 0:
                    blended = blend_overlap(tail_buffer, video_np, K)
                    output_video = np.concatenate([
                        blended, video_np[:, :, K:, :, :]
                    ], axis=2)
                    output_video = output_video[:, :, -clip_frames:, :, :]
                else:
                    output_video = video_np[:, :, -clip_frames:, :, :]

                if K > 0 and video_np.shape[2] >= K:
                    tail_buffer = video_np[:, :, -K:, :, :].copy()

            raw_clip_queue.put((output_video, audio))
        except Exception as e:
            print(f"[GEN] Drain: GPU {gpu_idx} failed: {e}")



# ---------------------------------------------------------------------------
# Worker 3: Post-processing (CPU)
# ---------------------------------------------------------------------------

def postprocess_thread(
    raw_clip_queue: queue.Queue,
    audio_out_queue: queue.Queue,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    print("[POST] Post-processing thread started.")
    while not stop_event.is_set():
        try:
            video_np, clip_audio = raw_clip_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.002)
            continue

        t0 = time.perf_counter()

        # 1. Push audio FIRST
        audio_msg = b'\x02' + clip_audio.tobytes()
        try:
            audio_out_queue.put_nowait(audio_msg)
        except queue.Full:
            try:
                audio_out_queue.get_nowait()
            except queue.Empty:
                pass
            audio_out_queue.put_nowait(audio_msg)

        # 2. Convert and push frames
        n_frames = video_np.shape[2]
        for f_idx in range(n_frames):
            if stop_event.is_set():
                return
            frame = video_np[0, :, f_idx, :, :]
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            frame = frame.transpose(1, 2, 0)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            _, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_msg = b'\x01' + jpeg.tobytes()

            try:
                frame_queue.put_nowait(frame_msg)
            except queue.Full:
                pass

        dt = time.perf_counter() - t0
        print(f"[POST] {n_frames} frames post-processed in {dt*1000:.1f}ms")


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def warmup_pipeline(gpu_manager: MultiGPUManager, ref_image, pose_provider, args):
    """Warmup all GPU pipelines."""
    dummy_audio = np.zeros(args.sample_rate, dtype=np.float32)

    for i in range(gpu_manager.num_gpus):
        pipe, device = gpu_manager.get_pipeline(i)
        ref_cache = gpu_manager.reference_caches[i]

        try:
            poses_tensor = pose_provider.get_batch(0, args.clip_frames)
            generator = torch.manual_seed(0)

            with torch.no_grad():
                pipe(
                    ref_image, dummy_audio,
                    poses_tensor[:, :, :args.clip_frames, ...],
                    args.width, args.height, args.clip_frames, args.steps, args.cfg,
                    generator=generator,
                    audio_sample_rate=args.sample_rate,
                    context_frames=12, fps=args.fps,
                    context_overlap=3, start_idx=0,
                    audio_margin=args.audio_margin,
                    init_latents=None,
                    reference_cache=ref_cache,
                    audio_context_frames=0,
                )
            print(f"[INIT] Warmup complete on {device}.")

        except Exception as e:
            print(f"[WARN] Warmup failed on {device}: {e}")
            import traceback; traceback.print_exc()
