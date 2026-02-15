import os
import time
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from moviepy.editor import VideoFileClip, AudioFileClip

# Import EchoMimic specific modules from your infer_acc.py
from diffusers import AutoencoderKL, DDIMScheduler
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2_acc import EchoMimicV2Pipeline 
from src.models.pose_encoder import PoseEncoder
from src.utils.util import save_videos_grid
from src.utils.dwpose_util import draw_pose_select_v2
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = str(Path(__file__).resolve().parent.parent)
INPUT_DIR = os.path.join(BASE_DIR, "pending_audio")
OUTPUT_DIR = os.path.join(BASE_DIR, "audio_video")
HEARTBEAT_FILE = os.path.join(BASE_DIR, "engine.heartbeat")
CONFIG_PATH = "./configs/prompts/infer_acc.yaml"

os.environ["FFMPEG_PATH"] = r"C:\ffmpeg\bin"
ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print("please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=./ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

os.makedirs(INPUT_DIR, exist_ok=True)

# --- GLOBAL MODEL STORAGE ---
pipe = None
weight_dtype = torch.float16
device = "cuda"

def load_models():
    global pipe, weight_dtype
    print("🚀 [WARM-UP] Initializing GPU Models...")
    config = OmegaConf.load(CONFIG_PATH)
    infer_config = OmegaConf.load(config.inference_config)
    
    vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path, local_files_only=True, # Forces offline mode
    torch_dtype=weight_dtype).to(device=device, dtype=weight_dtype)
    reference_unet = UNet2DConditionModel.from_pretrained(config.pretrained_base_model_path,
                    subfolder="unet").to(device=device, dtype=weight_dtype)
    reference_unet.load_state_dict(torch.load(config.reference_unet_path, map_location="cpu"))

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
    denoising_unet.load_state_dict(torch.load(config.denoising_unet_path, map_location="cpu"), strict=False)

    pose_net = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(device=device, dtype=weight_dtype)
    pose_net.load_state_dict(torch.load(config.pose_encoder_path))

    
    audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    pipe = EchoMimicV2Pipeline(
        vae=vae, reference_unet=reference_unet, denoising_unet=denoising_unet,
        audio_guider=audio_processor, pose_encoder=pose_net, scheduler=scheduler
    )
    pipe.to(device, dtype=weight_dtype)
    print("✅ [READY] Models are locked in VRAM.")

def update_heartbeat():
    with open(HEARTBEAT_FILE, "w") as f:
        f.write(str(time.time()))

class LocalAudioHandler(FileSystemEventHandler):
    def on_created(self, event):
        print(f"📢 Detected new file: {event.src_path}")
        if not event.is_directory and event.src_path.endswith(".wav"):
            self.process_video(event.src_path)

    def process_video(self, audio_path):
        try:
            print(f"🎬 Processing Audio: {os.path.basename(audio_path)}")
            # Logic from your infer_acc main loop
            # Using defaults for testing (can be expanded to read a sidecar JSON)
            L,H,W=240,768,768
            fps=24
            ref_image_path = "./assets/therapist_ref.png"
            pose_dir = "./assets/halfbody_demo/pose/01"
            
            ref_img_pil = Image.open(ref_image_path).convert("RGB")
            audio_clip = AudioFileClip(audio_path)
            L = min(L, int(audio_clip.duration * fps))

            # Generate Pose Tensor
            pose_list = []
            for i in range(L):
                tgt_musk_path = os.path.join(pose_dir, f"{i}.npy")
                detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
                imh, imw, rb, re, cb, ce = detected_pose['draw_pose_params']
                im = draw_pose_select_v2(detected_pose, imh, imw, ref_w=800)
                im = np.transpose(np.array(im),(1, 2, 0))
                
                mask = np.zeros((W, H, 3), dtype='uint8')
                mask[rb:re, cb:ce, :] = im
                pose_list.append(torch.Tensor(mask).to(dtype=weight_dtype, device=device).permute(2,0,1) / 255.0)

            poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)

            # Inference
            steps=6
            cfg=1.0
            seed=420
            generator = torch.manual_seed(seed)
            sample_rate = 16000
            context_overlap = 3
            video = pipe(ref_img_pil, audio_path, poses_tensor, W, H, L, steps, cfg, generator=generator, 
                         sample_rate=sample_rate, context_frames=12, fps=fps, 
                         context_overlap=context_overlap,start_idx=0).videos
            
            save_name = os.path.join(OUTPUT_DIR, os.path.basename(audio_path).replace(".wav", ""))

            save_videos_grid(video, save_name + "_temp.mp4", n_rows=1, fps=fps)
            
            # Combine Audio
            final_clip = VideoFileClip(save_name + "_temp.mp4")
            final_clip = final_clip.set_audio(audio_clip)
            final_clip.write_videofile(save_name + ".mp4", codec="libx264", audio_codec="aac", threads=2)
            
            os.remove(save_name + "_temp.mp4")
            os.remove(audio_path)
            print(f"🏁 Video Complete: {save_name}.mp4")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    load_models()
    observer = Observer()
    print(f"👀 Watching for new audio files in {INPUT_DIR}")
    observer.schedule(LocalAudioHandler(), INPUT_DIR, recursive=False)
    observer.start()
    try:
        while True:
            update_heartbeat()
            time.sleep(2)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()