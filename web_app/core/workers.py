
import os
import queue
import threading
import time
import cv2
import numpy as np
import torch
import random
from core.audio import _rms
from core.audio import _rms

# ---------------------------------------------------------------------------
# Worker 1: Input Preparation (CPU)
# ---------------------------------------------------------------------------

def input_preparation_thread(
    audio_queue, prepared_queue, stop_event,
    pose_provider,
    sample_rate, fps, clip_frames,
    vad_threshold=0.005, audio_margin=2,
):
    """
    Consumes raw audio chunks, performs VAD, prepares WAV/Pose inputs,
    and pushes `(audio_data, poses_tensor, ctx_frames, audio_chunk, reset_latent)` 
    to `prepared_queue`.
    """
    samples_per_clip = int(sample_rate * clip_frames / fps)
    
    # Audio Context Buffer: 2.0s history
    history_samples = sample_rate * 2
    audio_history_buffer = np.zeros(history_samples, dtype=np.float32)
    
    # Incoming accumulation buffer
    incoming_audio_buffer = np.array([], dtype=np.float32)

    pose_idx = 0
    consecutive_silent_clips = 0
    idle_clips_limit = max(1, int(1.0 * fps / clip_frames))

    print(f"[PREP] Waiting for audio (target clip: {samples_per_clip} samples) ...")
    print(f"[PREP] Audio Context: Rolling {2.0 + clip_frames/fps:.1f}s window (2.0s history + {clip_frames/fps:.1f}s new)")
    print(f"[PREP] Silence timeout: 1.0s (stop updating after {idle_clips_limit} silent clips)")
    
    # Pose provider is passed in arguments
    
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
                
                # Advance pose index (assuming provider knows full length? Provider handles wrapping itself?)
                # Wait, provider.get_batch(start_idx, num) handles wrapping logic internally
                # but we need to track our start_idx externally.
                # However, provider doesn't expose length in get_batch interface I defined?
                # Ah, I need to know how many poses there are to wrap pose_idx here if I want to display it
                # or just increment monotonically and let provider modulo it.
                # The provider's get_batch implementation does modulo. So just incrementing is fine.
                pose_idx = pose_idx + clip_frames

                # Push to GPU thread
                # Pass full_window (numpy array) directly as audio_data
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
# Worker 2: Video Generation (GPU)
# ---------------------------------------------------------------------------

def generate_video_clip(
    pipe, ref_image, audio_data, poses_tensor,
    W, H, clip_frames, sample_rate, fps,
    steps=4, cfg=1.0, init_latent=None, use_init_latent=True,
    audio_margin=2,
    reference_cache=None,
    audio_context_frames=0,
) -> tuple[np.ndarray | None, torch.Tensor | None]:
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
    pipe, ref_image, prepared_queue, raw_clip_queue, stop_event,
    reference_cache,
    sample_rate, fps, clip_frames, W, H, steps, cfg,
    use_init_latent=True, audio_margin=2,
):
    """
    Consumes prepared batches from `prepared_queue` and runs GPU inference.
    """
    last_latent = None

    while not stop_event.is_set():
        try:
            # Timeout allows checking stop_event
            batch = prepared_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.002)
            continue

        audio_data, poses_tensor, ctx_frames, current_audio, reset_latent = batch

        if reset_latent:
            last_latent = None
            print("[GEN] Latent reset due to silence.")

        try:
            t0 = time.perf_counter()
            video_np, final_latent = generate_video_clip(
                pipe, ref_image, audio_data, poses_tensor,
                W, H, clip_frames, sample_rate, fps, steps, cfg,
                init_latent=last_latent,
                use_init_latent=use_init_latent,
                audio_margin=audio_margin,
                reference_cache=reference_cache,
                audio_context_frames=ctx_frames,
            )
            
            if use_init_latent:
                last_latent = final_latent 

            dt = time.perf_counter() - t0
            print(f"[GEN] Clip generated in {dt:.2f}s")
            
            if video_np is not None:
                raw_clip_queue.put((video_np, current_audio))

        except Exception as e:
            print(f"[GEN] Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            pass


# ---------------------------------------------------------------------------
# Worker 3: Post-processing (CPU)
# ---------------------------------------------------------------------------

def postprocess_thread(
    raw_clip_queue, audio_out_queue, frame_queue, stop_event,
):
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

def warmup_pipeline(pipe, ref_image, pose_provider, args, reference_cache):
    # Create dummy Audio Array (1 second of silence)
    dummy_audio = np.zeros(args.sample_rate, dtype=np.float32)
    
    try:
        # Prepare dummy pose (using real files)
        # We use the first one
        poses_tensor = pose_provider.get_batch(0, args.clip_frames)
        
        # Generator
        generator = torch.manual_seed(0)
        
        # Run pipeline
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
                reference_cache=reference_cache,
                audio_context_frames=0, 
            )
        print("[INIT] Warmup complete. Compilation finished.")
        
    except Exception as e:
        print(f"[WARN] Warmup failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        pass
