
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

# Duration of silent audio to flush when the queue is empty (seconds)
EMPTY_AUDIO_DURATION: float = 0.5
# ---------------------------------------------------------------------------
# Worker 1: Input Preparation (CPU)
# ---------------------------------------------------------------------------

def _perform_idle_flush(
    audio_history_buffer: np.ndarray,
    history_samples: int,
    sample_rate: int,
    fps: int,
    pose_provider: PoseProvider,
    pose_idx: int,
    prepared_queue: queue.Queue,
) -> tuple[np.ndarray, int]:
    """Flushes a short silent clip to return avatar to neutral pose."""
    flush_samples = int(sample_rate * EMPTY_AUDIO_DURATION)
    print(f"[PREP] Queue empty. Flushing {EMPTY_AUDIO_DURATION}s of silent audio to reset position.")
    
    silent_clip = np.zeros(flush_samples, dtype=np.float32)
    # full_window = history + silence
    full_window = np.concatenate((audio_history_buffer, silent_clip))
    ctx_frames = int((len(audio_history_buffer) / sample_rate) * fps)
    
    try:
        # Get poses for this short silent clip
        flush_frames = int(EMPTY_AUDIO_DURATION * fps)
        poses_tensor = pose_provider.get_batch(pose_idx, flush_frames)
        new_pose_idx = pose_idx + flush_frames
        
        # Backlog is no longer passed here; calculated real-time in the loop
        batch = (full_window, poses_tensor, ctx_frames, silent_clip.copy(), False)
        prepared_queue.put(batch)
        
        # Update history with the silence we just "sent"
        new_history = np.concatenate((audio_history_buffer, silent_clip))[-history_samples:]
        return new_history, new_pose_idx
        
    except Exception as e:
        print(f"[PREP] Flush Error: {e}")
        return audio_history_buffer, pose_idx


def input_preparation_thread(
    audio_queue: queue.Queue,
    prepared_queue: queue.Queue,
    stop_event: threading.Event,
    pose_provider: PoseProvider,
    sample_rate: int,
    fps: int,
    clip_frames: int,
    active_clips: list[int],
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
    
    has_flushed = False
    should_reset_after_flush = False
    empty_flush_count = 0

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
            if consecutive_silent_clips > 0 or should_reset_after_flush:
                print(f"[PREP] Audio detected (RMS={clip_rms:.4f}). Starting...")
                # If we were silent for a long time, or we just flushed, reset latent.
                if (consecutive_silent_clips > (idle_clips_limit * 4)) or should_reset_after_flush:
                    reset_latent = True
                    if should_reset_after_flush:
                        print("[PREP] Resetting latent after empty flush.")
                    else:
                        print(f"[PREP] Silence limit hit ({consecutive_silent_clips} clips). Signaling pipeline reset.")
                
                should_reset_after_flush = False
            
            has_flushed = False
            consecutive_silent_clips = 0
            empty_flush_count = 0

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
                active_clips[0] += 1

                # CPU-side backlog: items in buffer + items in queue
                q_backlog = prepared_queue.qsize() * (clip_frames / fps)
                buf_backlog = len(incoming_audio_buffer) / sample_rate
                backlog_secs = buf_backlog + q_backlog
                print(f"[PREP] {backlog_secs:.2f}s of audio accumulated (CPU time).")

                # Update history
                audio_history_buffer = np.concatenate((audio_history_buffer, current_audio))[-history_samples:]

            except Exception as e:
                print(f"[PREP] Error: {e}")
                pass
        else:
            # 3. Handle empty queue / idle flush
            if audio_queue.empty() and not has_flushed and len(incoming_audio_buffer) < samples_per_clip:
                audio_history_buffer, pose_idx = _perform_idle_flush(
                    audio_history_buffer, history_samples, sample_rate, fps,
                    pose_provider, pose_idx, prepared_queue
                )
                has_flushed = True
                empty_flush_count += 1
                # Only reset after 3 consecutive empty/silent flushes
                if empty_flush_count >= 1:
                    should_reset_after_flush = True
                    empty_flush_count = 0
            
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
    if init_latent is not None and use_init_latent:
        print(f"[GEN]   (Pipeline) Using provided init_latent (Temporal Continuity ON)")
    else:
        print(f"[GEN]   (Pipeline) No init_latent used (Temporal Continuity OFF)")
    generator = torch.manual_seed(random.randint(0, 2**32 - 1))
    # Extend poses if shorter than clip_frames
    actual_pose_frames = poses_tensor.shape[2]
    if actual_pose_frames < clip_frames:
        pad_count = clip_frames - actual_pose_frames
        pad = poses_tensor[:, :, -1:, :, :].repeat(1, 1, pad_count, 1, 1)
        poses_tensor = torch.cat([poses_tensor, pad], dim=2)

    result = pipe(
        ref_image, audio_data,
        poses_tensor[:, :, :clip_frames, ...],
        W, H, clip_frames, steps, cfg,
        generator=generator,
        audio_sample_rate=sample_rate,
        context_frames=clip_frames, fps=fps,
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
    active_clips: Optional[list[int]] = None,
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
          f"overlap K={K}, clip_frames={clip_frames}, init_latent={use_init_latent}")

    if is_multi:
        _run_multi_gpu_loop(
            gpu_manager, ref_image, prepared_queue, raw_clip_queue, stop_event,
            sample_rate, fps, clip_frames, W, H, steps, cfg, audio_margin,
            num_gpus, K, active_clips
        )
    else:
        _run_single_gpu_loop(
            gpu_manager, ref_image, prepared_queue, raw_clip_queue, stop_event,
            sample_rate, fps, clip_frames, W, H, steps, cfg,
            use_init_latent, audio_margin, active_clips, K
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
    active_clips: list[int],
    K: int = 0,
) -> None:
    """Single-GPU: sequential generation with optional blend or latent continuity."""
    pipe, device = gpu_manager.get_pipeline(0)
    ref_cache = gpu_manager.reference_caches[0]

    is_first_chunk = True
    tail_buffer: Optional[np.ndarray] = None
    prev_current_audio: Optional[np.ndarray] = None

    samples_per_clip = int(sample_rate * clip_frames / fps)
    k_samples = int(sample_rate * K / fps)

    if K > 0:
        print(f"[GEN] Single-GPU Blending enabled (K={K}, Init-Latent disabled).")
        use_init_latent = False
    elif use_init_latent:
        print(f"[GEN] Latent continuity (init-latent) enabled.")
    else:
        print(f"[GEN] Latent continuity (init-latent) disabled.")

    while not stop_event.is_set():
        try:
            batch = prepared_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.002)
            continue

        audio_data, poses_tensor, ctx_frames, current_audio, reset_latent = batch

        if reset_latent:
            gpu_manager.last_latents[0] = None
            tail_buffer = None
            prev_current_audio = None
            is_first_chunk = True
            print("[GEN] GPU 0: Pipelines and latents reset due to silence/signal.")

        backlog_secs = prepared_queue.qsize() * (clip_frames / fps)
        print(f"[GEN] {backlog_secs:.2f}s of audio waiting (GPU time).")

        try:
            t0 = time.perf_counter()
            init_latent = gpu_manager.last_latents[0]

            # Determine frames and audio: ALWAYS use clip_frames + K if K > 0 
            # to keep shapes constant for torch.compile / CUDA Graphs.
            if K > 0:
                gen_frames = clip_frames + K
                if is_first_chunk:
                    # First chunk: pad audio with repeat to reach gen_frames
                    if len(audio_data) >= samples_per_clip:
                        # repeat the last samples_per_clip effectively (or just last bit)
                        k_audio_pad = audio_data[-k_samples:] if len(audio_data) >= k_samples else np.zeros(k_samples, dtype=np.float32)
                        final_audio = np.concatenate([audio_data, k_audio_pad])
                    else:
                        final_audio = audio_data
                    final_ctx = ctx_frames
                else:
                    # Subsequent: k_audio + current_audio
                    if prev_current_audio is not None and len(prev_current_audio) >= k_samples:
                        k_audio = prev_current_audio[-k_samples:]
                    else:
                        k_audio = np.zeros(k_samples, dtype=np.float32)

                    if len(audio_data) > samples_per_clip:
                        history_part = audio_data[:-samples_per_clip]
                        current_part = audio_data[-samples_per_clip:]
                    else:
                        history_part = np.array([], dtype=np.float32)
                        current_part = audio_data

                    final_audio = np.concatenate([history_part, k_audio, current_part])
                    final_ctx = ctx_frames
                
                prev_current_audio = current_audio.copy()
            else:
                gen_frames = clip_frames
                final_audio = audio_data
                final_ctx = ctx_frames

            # Retrieve previous latent if available and enabled
            latent_to_use = gpu_manager.last_latents[0] if use_init_latent else None
            if latent_to_use is not None:
                print(f"[GEN] GPU 0: Using previous latent for continuity.")

            video_np, final_latent = generate_video_clip(
                pipe, ref_image, final_audio, poses_tensor,
                W, H, gen_frames, sample_rate, fps, steps, cfg,
                init_latent=latent_to_use,
                use_init_latent=use_init_latent,
                audio_margin=audio_margin,
                reference_cache=ref_cache,
                audio_context_frames=final_ctx,
            )

            if use_init_latent and final_latent is not None:
                gpu_manager.last_latents[0] = final_latent
                print(f"[GEN] GPU 0: New latent stored for next chunk.")

            dt = time.perf_counter() - t0
            
            if video_np is not None:
                # Apply blending if K > 0
                if K > 0:
                    if is_first_chunk:
                        # First chunk: output only first clip_frames, save tail K
                        output_video = video_np[:, :, :clip_frames, :, :]
                        if video_np.shape[2] >= gen_frames:
                            tail_buffer = video_np[:, :, -K:, :, :].copy()
                        is_first_chunk = False
                    else:
                        if tail_buffer is not None:
                            blended = blend_overlap(tail_buffer, video_np, K)
                            output_video = np.concatenate([
                                blended,
                                video_np[:, :, K:gen_frames, :, :]
                            ], axis=2)
                            output_video = output_video[:, :, -clip_frames:, :, :]
                        else:
                            output_video = video_np[:, :, -clip_frames:, :, :]
                        
                        if video_np.shape[2] >= gen_frames:
                            tail_buffer = video_np[:, :, -K:, :, :].copy()
                else:
                    output_video = video_np

            print(f"[GEN] Clip generated in {dt:.2f}s ({output_video.shape[2]} frames)")
            raw_clip_queue.put((output_video, current_audio))

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

    video_np, _ = generate_video_clip(
        pipe, ref_image, audio_data,
        poses_tensor,
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
    active_clips: list[int],
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
                    print("[GEN] GPU 0-N: Pipelines and latents reset due to silence/signal.")

                # Real-time backlog: current prepared_queue + what is already dispatched (pending)
                q_backlog = prepared_queue.qsize() * (clip_frames / fps)
                p_backlog = len(pending) * (clip_frames / fps)
                backlog_secs = q_backlog + p_backlog
                print(f"[GEN] {backlog_secs:.2f}s of audio waiting (GPU time).")

                gpu_idx = next_gpu % num_gpus

                # ALWAYS use gen_frames = clip_frames + K if K > 0
                # to keep shapes constant for torch.compile / CUDA Graphs.
                if K > 0:
                    gen_frames = clip_frames + K
                    if is_first_chunk:
                        # Pad audio for the extra K frames
                        k_audio_pad = audio_data[-k_samples:] if len(audio_data) >= k_samples else np.zeros(k_samples, dtype=np.float32)
                        final_audio = np.concatenate([audio_data, k_audio_pad])
                        final_ctx = ctx_frames
                    else:
                        # audio_data = [history (2s)][current_audio (12 frames)]
                        # We insert K frames of PREVIOUS audio between them:
                        # extended  = [history (2s)][prev_k_audio (K frames)][current_audio (12 frames)]
                        # This gives the pipeline full context + 18 frames of real audio.

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
                else:
                    gen_frames = clip_frames
                    final_audio = audio_data
                    final_ctx = ctx_frames

                # Save current audio for next iteration's K overlap
                prev_current_audio = current_audio.copy()

                future = executor.submit(
                    _gpu_generate_task,
                    gpu_idx, gpu_manager, ref_image,
                    final_audio, poses_tensor,
                    gen_frames, W, H, sample_rate, fps, steps, cfg, audio_margin,
                    final_ctx,
                )
                pending.append((future, current_audio, gen_frames, gpu_idx, is_first_chunk, backlog_secs))

                # After first dispatch, subsequent chunks are no longer "first"
                if is_first_chunk:
                    is_first_chunk = False

                next_gpu += 1

            # ---- Process: check if the HEAD future is done (in-order) ----
            while pending and not stop_event.is_set():
                head_future, head_audio, head_gen_frames, head_gpu, head_was_first, head_backlog = pending[0]

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
                    # First chunk: output only first clip_frames, save tail K
                    if K > 0:
                        output_video = video_np[:, :, :clip_frames, :, :]
                    else:
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
                      f"{head_gen_frames} gen → {output_video.shape[2]} sent (Backlog: {head_backlog:.2f}s)")

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
        future, audio, gen_frames, gpu_idx, was_first, backlog = pending.popleft()
        try:
            video_np = future.result(timeout=GPU_RESULT_TIMEOUT_S)
            if video_np is None:
                continue

            if was_first:
                # First chunk: output only first clip_frames, save tail K
                if K > 0:
                    output_video = video_np[:, :, :clip_frames, :, :]
                else:
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

            print(f"[GEN] Drain: GPU {gpu_idx} - {gen_frames} gen (Backlog: {backlog:.2f}s)")
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
    active_clips: list[int],
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
        if active_clips:
            active_clips[0] = max(0, active_clips[0] - 1)
        print(f"[POST] {n_frames} frames post-processed in {dt*1000:.1f}ms (Remaining clips: {active_clips[0] if active_clips else '?'})")


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

            print(f"  [WARM] Warmup pass 1 (init_latents=None, ctx=0) on {device} ...")
            with torch.no_grad():
                result = pipe(
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
            
            # Pass 2: Warmup with init_latents and audio_context > 0
            # This captures the variations used in actual inference.
            if args.use_init_latent:
                print(f"  [WARM] Warmup pass 2 (init_latents=Tensor, ctx=12) on {device} ...")
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
                        init_latents=result.final_latent,
                        reference_cache=ref_cache,
                        audio_context_frames=12,
                    )

            print(f"[INIT] Warmup complete on {device}.")

        except Exception as e:
            print(f"[WARN] Warmup failed on {device}: {e}")
            import traceback; traceback.print_exc()
