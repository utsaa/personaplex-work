import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List
import time
import threading

import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model

from src.pipelines.pipeline_echomimicv2_acc import EchoMimicV2Pipeline 
from src.utils.util import get_fps, read_frames, save_videos_grid
from src.utils.dwpose_util import draw_pose_select_v2
import sys
from src.models.pose_encoder import PoseEncoder
from moviepy.editor import VideoFileClip, AudioFileClip

os.environ["FFMPEG_PATH"] = r"C:\ffmpeg\bin"
ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print("please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=./ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"


# ---------------------------------------------------------------------------
# Multi-GPU Utilities
# ---------------------------------------------------------------------------

def detect_gpus() -> int:
    """Return number of available CUDA GPUs."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def blend_overlap(tail_video: np.ndarray, head_video: np.ndarray, K: int) -> np.ndarray:
    """Linear crossfade between last K frames of tail and first K frames of head.
    
    Both inputs: shape (1, C, F, H, W), values in [0, 1].
    Returns blended array (1, C, K, H, W).
    """
    if K <= 0:
        return np.empty((tail_video.shape[0], tail_video.shape[1], 0,
                         tail_video.shape[3], tail_video.shape[4]),
                        dtype=tail_video.dtype)
    tail_slice = tail_video[:, :, -K:, :, :]
    head_slice = head_video[:, :, :K, :, :]
    alpha = np.linspace(0.0, 1.0, K, dtype=np.float32).reshape(1, 1, K, 1, 1)
    return (1.0 - alpha) * tail_slice + alpha * head_slice


def load_pipeline_on_device(config, infer_config, device, weight_dtype, args):
    """Load a full EchoMimic-v2 pipeline on a specific CUDA device."""
    print(f"[INIT] Loading pipeline on {device} ...")
    
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
        local_files_only=True,
        torch_dtype=weight_dtype
    ).to(device, dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )

    if os.path.exists(config.motion_module_path):
        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device=device)
    else:
        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
                "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
            }
        ).to(dtype=weight_dtype, device=device)
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False
    )

    pose_net = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=device
    )
    pose_net.load_state_dict(torch.load(config.pose_encoder_path))

    adapter_path = None
    if args.audio_model_type == "wav2vec2":
        audio_model_path = getattr(config, "wav2vec2_audio_guider_path", config.audio_model_path)
        adapter_path = getattr(config, "wav2vec2_audio_adapter_path", "./pretrained_weights/audio_processor/wav2vec2/trained_adapter.pt")
    else:
        audio_model_path = config.audio_model_path
        
    audio_processor = load_audio_model(model_path=audio_model_path, device=device, model_type=args.audio_model_type, adapter_path=adapter_path)

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
    print(f"[INIT] Pipeline on {device} ready.")
    return pipe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/prompts/infer_acc.yaml")
    parser.add_argument("-W", type=int, default=768)
    parser.add_argument("-H", type=int, default=768)
    parser.add_argument("-L", type=int, default=240)
    parser.add_argument("--seed", type=int, default=420)

    parser.add_argument("--context_frames", type=int, default=12)
    parser.add_argument("--context_overlap", type=int, default=3)
   
    parser.add_argument("--motion_sync", type=int, default=1)

    parser.add_argument("--output_video_dir_path", type=str, default="./output/output_videos")

    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--ref_images_dir", type=str, default=f'./assets/halfbody_demo/refimag')
    parser.add_argument("--audio_dir", type=str, default='./assets/halfbody_demo/audio')
    parser.add_argument("--pose_dir", type=str, default="./assets/halfbody_demo/pose")
    parser.add_argument("--refimg_name", type=str, default='natural_bk_openhand/0035.png')
    parser.add_argument("--audio_name", type=str, default='chinese/echomimicv2_woman.wav')
    parser.add_argument("--pose_name", type=str, default="01")
    parser.add_argument("--audio_model_type", type=str, default="whisper", choices=["whisper", "wav2vec2"])
    parser.add_argument("--overlap_frames", type=int, default=12,
                        help="Number of overlap frames (K) per GPU boundary for blending. Default: 12")

    args = parser.parse_args()

    return args


def generate_chunk(pipe, ref_img_pil, audio_path, poses_tensor,
                   width, height, num_frames, args, generator, device):
    """Run pipeline for a chunk of frames. Returns video numpy (1,C,F,H,W)."""
    video = pipe(
        ref_img_pil,
        audio_path,
        poses_tensor[:, :, :num_frames, ...],
        width,
        height,
        num_frames,
        args.steps,
        args.cfg,
        generator=generator,
        audio_sample_rate=args.sample_rate,
        context_frames=12,
        fps=args.fps,
        context_overlap=args.context_overlap,
        start_idx=0
    ).videos
    
    if isinstance(video, torch.Tensor):
        return video.cpu().numpy()
    return video


def main():
    args = parse_args()

    config = OmegaConf.load(args.config)
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)

    # ---------------------------------------------------------------------------
    # Detect GPUs
    # ---------------------------------------------------------------------------
    N = detect_gpus()
    if N == 0:
        print("[WARN] No CUDA GPUs found, falling back to CPU.")
        N = 1
        devices = ["cpu"]
    else:
        print(f"[INIT] Detected {N} CUDA GPU(s).")
        devices = [f"cuda:{i}" for i in range(N)]

    K = args.overlap_frames if N >= 2 else 0
    print(f"[INIT] Using {N} GPU(s), overlap K={K} frames per boundary.")

    # ---------------------------------------------------------------------------
    # Load N pipelines (in parallel for multi-GPU)
    # ---------------------------------------------------------------------------
    pipes = [None] * N
    errors = [None] * N

    if N == 1:
        pipes[0] = load_pipeline_on_device(config, infer_config, devices[0], weight_dtype, args)
    else:
        def _load(idx):
            try:
                pipes[idx] = load_pipeline_on_device(config, infer_config, devices[idx], weight_dtype, args)
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
                raise RuntimeError(f"Failed to load pipeline on {devices[i]}: {err}")

    print(f"[INIT] All {N} pipeline(s) loaded.")

    # ---------------------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------------------
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--step_{args.steps}-{args.W}x{args.H}--cfg_{args.cfg}"
    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    for ref_image_path in config["test_cases"].keys():
        for file_path in config["test_cases"][ref_image_path]:
            if ".wav" in file_path:
                audio_path = file_path
            else:
                pose_dir = file_path

        if args.seed is not None and args.seed > -1:
            generator = torch.manual_seed(args.seed)
        else:
            generator = torch.manual_seed(random.randint(100, 1000000))

        ref_name = Path(ref_image_path).stem
        audio_name = Path(audio_path).stem
        final_fps = args.fps

        inputs_dict = {
            "refimg": f'{ref_image_path}',
            "audio": f'{audio_path}',
            "pose": f'{pose_dir}',
        }

        start_idx = 0

        print('Pose:', inputs_dict['pose'])
        print('Reference:', inputs_dict['refimg'])
        print('Audio:', inputs_dict['audio'])

        save_path = Path(f"{save_dir}/{ref_name}")    
        save_path.mkdir(exist_ok=True, parents=True)
        save_name = f"{save_path}/{ref_name}-a-{audio_name}-i{start_idx}"

        ref_img_pil = Image.open(ref_image_path).convert("RGB")
        audio_clip = AudioFileClip(inputs_dict['audio'])
    
        total_frames = min(args.L, int(audio_clip.duration * final_fps), len(os.listdir(inputs_dict['pose'])))
        args.L = total_frames

        # ==================== Prepare all poses ====================
        pose_list_all = []
        for index in range(start_idx, start_idx + total_frames + K * (N - 1)):
            # Clamp index to available poses
            actual_idx = min(index, len(os.listdir(inputs_dict['pose'])) - 1)
            tgt_musk = np.zeros((args.W, args.H, 3)).astype('uint8')
            tgt_musk_path = os.path.join(inputs_dict['pose'], "{}.npy".format(actual_idx))
            detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
            imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']
            im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
            im = np.transpose(np.array(im),(1, 2, 0))
            tgt_musk[rb:re,cb:ce,:] = im
            tgt_musk_pil = Image.fromarray(np.array(tgt_musk)).convert('RGB')
            pose_list_all.append(torch.Tensor(np.array(tgt_musk_pil)).permute(2,0,1) / 255.0)

        # ==================== Generate (single or multi-GPU) ====================
        audio_clip_full = AudioFileClip(inputs_dict['audio'])
        
        if N == 1:
            # ------- Single GPU: original behavior -------
            print(f"[GEN] Single GPU: generating {total_frames} frames on {devices[0]}...")
            device = devices[0]
            poses_for_gpu = [p.to(dtype=weight_dtype, device=device) for p in pose_list_all[:total_frames]]
            poses_tensor = torch.stack(poses_for_gpu, dim=1).unsqueeze(0)
            
            audio_clip_gen = audio_clip_full.set_duration(total_frames / final_fps)

            t0 = time.perf_counter()
            video = generate_chunk(
                pipes[0], ref_img_pil, inputs_dict['audio'],
                poses_tensor, args.W, args.H, total_frames, args, generator, device
            )
            dt = time.perf_counter() - t0
            print(f"[GEN] Generated in {dt:.2f}s")

        else:
            # ------- Multi-GPU: Parallel with overlap blend -------
            # Calculate chunk boundaries
            base_chunk = total_frames // N
            remainder = total_frames % N

            # chunks[i] = (output_start, output_end, gen_start, gen_end)
            # output_start/end = frames that end up in final video
            # gen_start/end = frames actually generated (includes overlap)
            chunks = []
            pos = 0
            for i in range(N):
                chunk_len = base_chunk + (1 if i < remainder else 0)
                out_start = pos
                out_end = pos + chunk_len

                # Gen range includes K overlap on each side (except first/last GPU)
                gen_start = max(0, out_start - K) if i > 0 else out_start
                gen_end = min(total_frames + K, out_end + K) if i < N - 1 else out_end

                chunks.append((out_start, out_end, gen_start, gen_end))
                pos = out_end

            print(f"[GEN] Multi-GPU ({N} GPUs, K={K}) chunk plan:")
            for i, (os_, oe, gs, ge) in enumerate(chunks):
                print(f"  GPU {i}: output [{os_}:{oe}] ({oe-os_} frames), "
                      f"generate [{gs}:{ge}] ({ge-gs} frames)")

            # Launch all GPUs in parallel
            results = [None] * N
            gen_errors = [None] * N

            def _generate(gpu_i):
                try:
                    out_start, out_end, gen_start, gen_end = chunks[gpu_i]
                    device = devices[gpu_i]
                    gen_frames = gen_end - gen_start

                    # Prepare poses for this chunk
                    poses_for_gpu = [
                        pose_list_all[j].to(dtype=weight_dtype, device=device)
                        for j in range(gen_start, min(gen_end, len(pose_list_all)))
                    ]
                    # Pad if needed
                    while len(poses_for_gpu) < gen_frames:
                        poses_for_gpu.append(poses_for_gpu[-1])
                    poses_tensor = torch.stack(poses_for_gpu, dim=1).unsqueeze(0)

                    # Use a different seed per GPU for noise diversity
                    gpu_gen = torch.manual_seed(args.seed + gpu_i * 1000)

                    t0 = time.perf_counter()
                    video_np = generate_chunk(
                        pipes[gpu_i], ref_img_pil, inputs_dict['audio'],
                        poses_tensor, args.W, args.H, gen_frames, args, gpu_gen, device
                    )
                    dt = time.perf_counter() - t0
                    print(f"[GEN] GPU {gpu_i}: {gen_frames} frames in {dt:.2f}s")
                    results[gpu_i] = video_np
                except Exception as e:
                    gen_errors[gpu_i] = e

            t0_total = time.perf_counter()
            threads = []
            for i in range(N):
                t = threading.Thread(target=_generate, args=(i,))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
            dt_total = time.perf_counter() - t0_total

            for i, err in enumerate(gen_errors):
                if err is not None:
                    raise RuntimeError(f"Generation failed on GPU {i}: {err}")

            print(f"[GEN] All GPUs finished in {dt_total:.2f}s (parallel)")

            # Assemble final video with overlap blending
            video_parts = []
            for i in range(N):
                out_start, out_end, gen_start, gen_end = chunks[i]
                chunk_video = results[i]  # (1, C, gen_frames, H, W)

                if i == 0:
                    # First chunk: take frames up to out_end, keep overlap tail
                    # out_end - gen_start = frames from this chunk that are "ours"
                    # But we also generated K extra frames at the end for blending
                    keep_end = out_end - gen_start  # index within chunk_video
                    video_parts.append(chunk_video[:, :, :keep_end, :, :])
                elif i == N - 1:
                    # Last chunk: blend first K frames with previous chunk's tail
                    prev_out_start, prev_out_end, prev_gen_start, prev_gen_end = chunks[i - 1]
                    prev_video = results[i - 1]
                    
                    # Previous chunk's tail K frames (the overlap zone)
                    prev_tail = prev_video[:, :, -K:, :, :]
                    # Current chunk's first K frames (the overlap zone)
                    curr_head = chunk_video[:, :, :K, :, :]
                    
                    blended = blend_overlap(prev_tail, curr_head, K)
                    
                    # Final: blended + rest of current chunk
                    video_parts.append(blended)
                    video_parts.append(chunk_video[:, :, K:, :, :])
                else:
                    # Middle chunk: blend start, include middle, keep tail for next
                    prev_video = results[i - 1]
                    prev_tail = prev_video[:, :, -K:, :, :]
                    curr_head = chunk_video[:, :, :K, :, :]
                    
                    blended = blend_overlap(prev_tail, curr_head, K)
                    
                    keep_end = out_end - gen_start
                    video_parts.append(blended)
                    video_parts.append(chunk_video[:, :, K:keep_end, :, :])

            video = np.concatenate(video_parts, axis=2)
            print(f"[GEN] Assembled video: {video.shape[2]} frames")

        # ==================== Save output ====================
        final_length = min(video.shape[2], total_frames)
        video_sig = video[:, :, :final_length, :, :]
        
        # Convert to torch tensor if numpy
        if isinstance(video_sig, np.ndarray):
            video_sig = torch.from_numpy(video_sig)
        
        save_videos_grid(
            video_sig,
            save_name + "_woa_sig.mp4",
            n_rows=1,
            fps=final_fps,
        )

        audio_clip_out = AudioFileClip(inputs_dict['audio'])
        audio_clip_out = audio_clip_out.set_duration(final_length / final_fps)
        
        video_clip_sig = VideoFileClip(save_name + "_woa_sig.mp4",)
        video_clip_sig = video_clip_sig.set_audio(audio_clip_out)
        video_clip_sig.write_videofile(save_name + "_sig.mp4", codec="libx264", audio_codec="aac", threads=2)
        output_video_dir = args.output_video_dir_path
        if output_video_dir and os.path.exists(output_video_dir):
            video_clip_sig.write_videofile(output_video_dir + ".mp4", codec="libx264", audio_codec="aac", threads=2)
        else:
            print(f"Output video dir path {output_video_dir} does not exist {os.path.exists(output_video_dir)}. Skipping copy to output dir.")
        
        # Clean up temp file (use platform-appropriate command)
        try:
            os.remove(save_name + "_woa_sig.mp4")
        except OSError:
            pass
        print(save_name)

if __name__ == "__main__":
    main()
