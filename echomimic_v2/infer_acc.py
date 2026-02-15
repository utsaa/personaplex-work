import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List
import time

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

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    config = OmegaConf.load(args.config)
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    device = args.device
    if device.__contains__("cuda") and not torch.cuda.is_available():
        device = "cpu"

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)

    ############# model_init started #############

    ## vae init
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
        local_files_only=True, # Forces offline mode
        torch_dtype=weight_dtype
    ).to("cuda", dtype=weight_dtype)

    ## reference net init
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )

    ## denoising net init
    if os.path.exists(config.motion_module_path):
        ### stage1 + stage2
        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device=device)
    else:
        ### only stage1
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

    ## face locator init
    pose_net = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )
    pose_net.load_state_dict(torch.load(config.pose_encoder_path))

    ### load audio processor params
    audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

    ############# model_init finished #############

    width, height = args.W, args.H
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

    pipe = pipe.to("cuda", dtype=weight_dtype)

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
    
        args.L = min(args.L, int(audio_clip.duration * final_fps), len(os.listdir(inputs_dict['pose'])))  
        # ==================== face_locator =====================
        pose_list = []
        for index in range(start_idx, start_idx + args.L):
            tgt_musk = np.zeros((args.W, args.H, 3)).astype('uint8')
            tgt_musk_path = os.path.join(inputs_dict['pose'], "{}.npy".format(index))
            detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
            imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']
            im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
            im = np.transpose(np.array(im),(1, 2, 0))
            tgt_musk[rb:re,cb:ce,:] = im

            tgt_musk_pil = Image.fromarray(np.array(tgt_musk)).convert('RGB')
            pose_list.append(torch.Tensor(np.array(tgt_musk_pil)).to(dtype=weight_dtype, device=device).permute(2,0,1) / 255.0)

        poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
        audio_clip = AudioFileClip(inputs_dict['audio'])
        audio_clip = audio_clip.set_duration(args.L / final_fps)

        video = pipe(
            ref_img_pil,
            inputs_dict['audio'],
            poses_tensor[:,:,:args.L,...],
            width,
            height,
            args.L,
            args.steps,
            args.cfg,
            generator=generator,
            audio_sample_rate=args.sample_rate,
            context_frames=12,
            fps=final_fps,
            context_overlap=args.context_overlap,
            start_idx=start_idx
        ).videos

        final_length = min(video.shape[2], poses_tensor.shape[2], args.L)

        video_sig = video[:, :, :final_length, :, :]
        save_videos_grid(
            video_sig,
            save_name + "_woa_sig.mp4",
            n_rows=1,
            fps=final_fps,
        )

        video_clip_sig = VideoFileClip(save_name + "_woa_sig.mp4",)
        video_clip_sig = video_clip_sig.set_audio(audio_clip)
        video_clip_sig.write_videofile(save_name + "_sig.mp4", codec="libx264", audio_codec="aac", threads=2)
        output_video_dir = args.output_video_dir_path
        if output_video_dir and os.path.exists(output_video_dir):
            video_clip_sig.write_videofile(output_video_dir + ".mp4", codec="libx264", audio_codec="aac", threads=2)
        else:
            print(f"Output video dir path {output_video_dir} does not exist {os.path.exists(output_video_dir)}. Skipping copy to output dir.")
        os.system("rm {}".format(save_name + "_woa_sig.mp4"))
        print(save_name)

if __name__ == "__main__":
    main()

